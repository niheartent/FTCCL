/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>
#include <float.h>
#include <libgen.h>
#include <dirent.h> 
#include "core.h"
#include "nvmlwrap.h"
#include "xml.h"
#include "rocm_smi_wrap.h"
#include "archinfo.h"
#include "comm.h"
#include "param.h"
#include <fstream>
#if defined(__x86_64__)
#include <cpuid.h>
#endif

static bool isTopoMappingDefaultFile = true;
static struct ncclBusIds  g_busIdSet;
std::once_flag envTypeInfo::init_flag;
envTypeInfo*   envTypeInfo::_inst = NULL;
/*******************/
/* XML File Parser */
/*******************/

ncclResult_t xmlGetChar(FILE* file, char* c) {
  if (fread(c, 1, 1, file) == 0) {
    WARN("XML Parse : Unexpected EOF");
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t xmlGetValue(FILE* file, char* value, char* last) {
  char c;
  NCCLCHECK(xmlGetChar(file, &c));
  if (c != '"' && c != '\'') {
#if INT_OK
    int o = 0;
    do {
      value[o++] = c;
      NCCLCHECK(xmlGetChar(file, &c));
    } while (c >= '0' && c <= '9');
    value[o] = '\0';
    *last = c;
    return ncclSuccess;
#else
    WARN("XML Parse : Expected (double) quote.");
    return ncclInternalError;
#endif
  }
  int o = 0;
  do {
    NCCLCHECK(xmlGetChar(file, &c));
    value[o++] = c;
  } while (c != '"');
  value[o-1] = '\0';
  NCCLCHECK(xmlGetChar(file, last));
  return ncclSuccess;
}

ncclResult_t xmlGetToken(FILE* file, char* name, char* value, char* last) {
  char c;
  char* ptr = name;
  int o = 0;
  do {
    NCCLCHECK(xmlGetChar(file, &c));
    if (c == '=') {
      ptr[o] = '\0';
      if (value == NULL) {
        WARN("XML Parse : Unexpected value with name %s", ptr);
        return ncclInternalError;
      }
      return xmlGetValue(file, value, last);
    }
    ptr[o] = c;
    if (o == MAX_STR_LEN-1) {
      ptr[o] = '\0';
      WARN("Error : name %s too long (max %d)", ptr, MAX_STR_LEN);
      return ncclInternalError;
    }
    o++;
  } while (c != ' ' && c != '>' && c != '/' && c != '\n' && c != '\r');
  ptr[o-1] = '\0';
  *last = c;
  return ncclSuccess;
}

// Shift the 3-chars string by one char and append c at the end
#define SHIFT_APPEND(s, c) do { s[0]=s[1]; s[1]=s[2]; s[2]=c; } while(0)
ncclResult_t xmlSkipComment(FILE* file, char* start, char next) {
  // Start from something neutral with \0 at the end.
  char end[4] = "...";

  // Inject all trailing chars from previous reads. We don't need
  // to check for --> here because there cannot be a > in the name.
  for (int i=0; i<strlen(start); i++) SHIFT_APPEND(end, start[i]);
  SHIFT_APPEND(end, next);

  // Stop when we find "-->"
  while (strcmp(end, "-->") != 0) {
    int c;
    if (fread(&c, 1, 1, file) != 1) {
      WARN("XML Parse error : unterminated comment");
      return ncclInternalError;
    }
    SHIFT_APPEND(end, c);
  }
  return ncclSuccess;
}

ncclResult_t xmlGetNode(FILE* file, struct ncclXmlNode* node) {
  node->type = NODE_TYPE_NONE;
  char c = ' ';
  while (c == ' ' || c == '\n' || c == '\r') {
    if (fread(&c, 1, 1, file) == 0) return ncclSuccess;
  }
  if (c != '<') {
    WARN("XML Parse error : expecting '<', got '%c'", c);
    return ncclInternalError;
  }
  // Read XML element name
  NCCLCHECK(xmlGetToken(file, node->name, NULL, &c));

  // Check for comments
  if (strncmp(node->name, "!--", 3) == 0) {
    NCCLCHECK(xmlSkipComment(file, node->name+3, c));
    return xmlGetNode(file, node);
  }

  // Check for closing tag
  if (node->name[0] == '\0' && c == '/') {
    node->type = NODE_TYPE_CLOSE;
    // Re-read the name, we got '/' in the first call
    NCCLCHECK(xmlGetToken(file, node->name, NULL, &c));
    if (c != '>') {
      WARN("XML Parse error : unexpected trailing %c in closing tag %s", c, node->name);
      return ncclInternalError;
    }
    return ncclSuccess;
  }

  node->type = NODE_TYPE_OPEN;

  // Get Attributes
  int a = 0;
  while (c == ' ') {
    NCCLCHECK(xmlGetToken(file, node->attrs[a].key, node->attrs[a].value, &c));
    if (a == MAX_ATTR_COUNT) {
      INFO(NCCL_GRAPH, "XML Parse : Ignoring extra attributes (max %d)", MAX_ATTR_COUNT);
      // Actually we need to still consume the extra attributes so we have an extra one.
    } else a++;
  }
  node->nAttrs = a;
  if (c == '/') {
    node->type = NODE_TYPE_SINGLE;
    char str[MAX_STR_LEN];
    NCCLCHECK(xmlGetToken(file, str, NULL, &c));
  }
  if (c != '>') {
    WARN("XML Parse : expected >, got '%c'", c);
    return ncclInternalError;
  }
  return ncclSuccess;
}

typedef ncclResult_t (*xmlHandlerFunc_t)(FILE*, struct ncclXml*, struct ncclXmlNode*);

struct xmlHandler {
  const char * name;
  xmlHandlerFunc_t func;
};

ncclResult_t xmlLoadSub(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head, struct xmlHandler handlers[], int nHandlers) {
  if (head && head->type == NODE_TYPE_SINGLE) return ncclSuccess;
  while (1) {
    if (xml->maxIndex == xml->maxNodes) {
      WARN("Error : XML parser is limited to %d nodes", xml->maxNodes);
      return ncclInternalError;
    }
    struct ncclXmlNode* node = xml->nodes+xml->maxIndex;
    memset(node, 0, sizeof(struct ncclXmlNode));
    NCCLCHECK(xmlGetNode(file, node));
    if (node->type == NODE_TYPE_NONE) {
      if (head) {
        WARN("XML Parse : unterminated %s", head->name);
        return ncclInternalError;
      } else {
        // All done
        return ncclSuccess;
      }
    }
    if (head && node->type == NODE_TYPE_CLOSE) {
      if (strcmp(node->name, head->name) != 0) {
        WARN("XML Mismatch : %s / %s", head->name, node->name);
        return ncclInternalError;
      }
      return ncclSuccess;
    }
    int found = 0;
    for (int h=0; h<nHandlers; h++) {
      if (strcmp(node->name, handlers[h].name) == 0) {
        if (head) {
          if (head->nSubs == MAX_SUBS) {
            WARN("Error : XML parser is limited to %d subnodes", MAX_SUBS);
            return ncclInternalError;
          }
          head->subs[head->nSubs++] = node;
        }
        node->parent = head;
        node->nSubs = 0;
        xml->maxIndex++;
        NCCLCHECK(handlers[h].func(file, xml, node));
        found = 1;
        break;
      }
    }
    if (!found) {
      if (nHandlers) INFO(NCCL_GRAPH, "Ignoring element %s", node->name);
      NCCLCHECK(xmlLoadSub(file, xml, node, NULL, 0));
    }
  }
}

/**************/
/* XML Writer */
/**************/

// exp == 1 -- serialize; exp == 0 -- deserialize
ncclResult_t ncclTopoConvertXml(struct ncclXml* xml, uintptr_t base, int exp) {
  for (int n = 0; n < xml->maxIndex; n++) {
    struct ncclXmlNode *node = &xml->nodes[n];

    // For "parent", we shift the base by 1 so that we can distinguish actual
    // NULL pointers from pointers pointing to the first node.
    if (node->parent)
      node->parent = (struct ncclXmlNode *) (exp ? ((uintptr_t)node->parent - base + 1) : (base - 1 + (uintptr_t)node->parent));

    for (int s = 0; s < node->nSubs; s++) {
      node->subs[s] = (struct ncclXmlNode *) (exp ? ((uintptr_t)node->subs[s] - base) : (base + (uintptr_t)node->subs[s]));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDumpXmlRec(int indent, FILE* file, struct ncclXmlNode* node) {
  for (int i=0; i<indent; i++) fprintf(file, " ");
  fprintf(file, "<%s", node->name);

  for (int a=0; a<node->nAttrs; a++) {
    fprintf(file, " %s=\"%s\"", node->attrs[a].key, node->attrs[a].value);
  }
  if (node->nSubs == 0) {
    fprintf(file, "/>\n");
  } else {
    fprintf(file, ">\n");
    for (int s=0; s<node->nSubs; s++) {
      NCCLCHECK(ncclTopoDumpXmlRec(indent+2, file, node->subs[s]));
    }
    for (int i=0; i<indent; i++) fprintf(file, " ");
    fprintf(file, "</%s>\n", node->name);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDumpXmlToFile(const char* xmlTopoFile, struct ncclXml* xml) {
  FILE* file = fopen(xmlTopoFile, "w");
  if (file == NULL) {
    WARN("Unable to open %s, not dumping topology.", xmlTopoFile);
    return ncclSuccess;
  }
  NCCLCHECK(ncclTopoDumpXmlRec(0, file, xml->nodes));
  fclose(file);
  return ncclSuccess;
}

static ncclResult_t xmlTopoFuseXmlRecursive(struct ncclXml* dst, struct ncclXmlNode* dstParent, struct ncclXmlNode* srcParent) {
  for (int i = 0; i < srcParent->nSubs; i++) {
    struct ncclXmlNode* srcNode = srcParent->subs[i];
    struct ncclXmlNode* dstNode;
    NCCLCHECK(xmlFindNode(dstParent, srcNode, &dstNode));
    if (dstNode == NULL) {
      NCCLCHECK(xmlAddTree(dst, dstParent, srcNode));
    } else {
      NCCLCHECK(xmlTopoFuseXmlRecursive(dst, dstNode, srcNode));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoFuseXml(struct ncclXml* dst, struct ncclXml* src) {
  struct ncclXmlNode* topNodeDst;
  NCCLCHECK(xmlFindTag(dst, "system", &topNodeDst));

  if (topNodeDst == NULL) {
    xmlAddTree(dst, NULL, src->nodes);
    return ncclSuccess;
  }

  struct ncclXmlNode* topNodeSrc;
  NCCLCHECK(xmlFindTag(src, "system", &topNodeSrc));

  NCCLCHECK(xmlTopoFuseXmlRecursive(dst, topNodeDst, topNodeSrc));

  return ncclSuccess;
}


/****************************************/
/* Parser rules for our specific format */
/****************************************/

ncclResult_t ncclTopoXmlLoadNvlink(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPciLink(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadC2c(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}
ncclResult_t ncclTopoXmlLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  struct xmlHandler handlers[] = { { "xgmi", ncclTopoXmlLoadNvlink } };
#else
  struct xmlHandler handlers[] = { { "nvlink", ncclTopoXmlLoadNvlink }, { "c2c", ncclTopoXmlLoadC2c } };
#endif
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNic(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "net", ncclTopoXmlLoadNet } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPci(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "pci", ncclTopoXmlLoadPci }, { "gpu", ncclTopoXmlLoadGpu }, { "nic", ncclTopoXmlLoadNic}, { "pcilink", ncclTopoXmlLoadPciLink} };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 4));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadCpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "pci", ncclTopoXmlLoadPci }, { "nic", ncclTopoXmlLoadNic } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadSystem(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  int version;
  NCCLCHECK(xmlGetAttrInt(head, "version", &version));
  if (version != NCCL_TOPO_XML_VERSION) {
    WARN("XML Topology has wrong version %d, %d needed", version, NCCL_TOPO_XML_VERSION);
    return ncclInvalidUsage;
  }
  const char* name;
  NCCLCHECK(xmlGetAttr(head, "name", &name));
  if (name != NULL) {
    INFO(NCCL_GRAPH, "Loading topology %s", name);
  } else {
    INFO(NCCL_GRAPH, "Loading unnamed topology");
  }

  struct xmlHandler handlers[] = { { "cpu", ncclTopoXmlLoadCpu } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromFile(const char* xmlTopoFile, struct ncclXml* xml, int warn) {
  FILE* file = fopen(xmlTopoFile, "r");
  if (file == NULL) {
    if (warn) {
      WARN("Could not open XML topology file %s : %s", xmlTopoFile, strerror(errno));
    }
    return ncclSuccess;
  }
  INFO(NCCL_GRAPH, "Loading topology file %s", xmlTopoFile);
  struct xmlHandler handlers[] = { { "system", ncclTopoXmlLoadSystem } };
  xml->maxIndex = 0;
  NCCLCHECK(xmlLoadSub(file, xml, NULL, handlers, 1));
  fclose(file);
  return ncclSuccess;
}

/**********************/
/* XML creation       */
/* from autodetection */
/**********************/

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))
static void memcpylower(char* dst, const char* src, const size_t size) {
  for (int i=0; i<size; i++) dst[i] = tolower(src[i]);
}
static ncclResult_t getPciPath(const char* busId, char** path) {
  char busPath[] = "/sys/class/pci_bus/0000:00/../../0000:00:00.0";
  memcpylower(busPath+sizeof("/sys/class/pci_bus/")-1, busId, BUSID_REDUCED_SIZE-1);
  memcpylower(busPath+sizeof("/sys/class/pci_bus/0000:00/../../")-1, busId, BUSID_SIZE-2);
  // override PCIe device function ID in CPX mode
  busPath[sizeof("/sys/class/pci_bus/0000:00/../../")+BUSID_SIZE-3] = '0';
  *path = realpath(busPath, NULL);
  if (*path == NULL) {
    WARN("Could not find real path of %s", busPath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

#include <dirent.h>
static ncclResult_t getBcmLinks(const char* busId, int* nlinks, char** peers) {
  *nlinks = 0;
  *peers = NULL;
  char dirPath[] = "/sys/kernel/pci_switch_link/virtual_switch_links/0000:00:00.0";
  memcpylower(dirPath+sizeof("/sys/kernel/pci_switch_link/virtual_switch_links/")-1, busId, BUSID_SIZE-1);
  DIR *dir = opendir(dirPath);
  if (dir) {
    struct dirent* file;
    while ((file = readdir(dir)) != NULL) {
      if (strlen(file->d_name) != BUSID_SIZE-1) continue;
      char* path;
      if (getPciPath(file->d_name, &path) == ncclSystemError) continue;
      free(path);
      NCCLCHECK(ncclRealloc(peers, (*nlinks)*BUSID_SIZE, ((*nlinks)+1)*BUSID_SIZE));
      memcpy((*peers)+BUSID_SIZE*(*nlinks)++, file->d_name, BUSID_SIZE);
    }
    closedir(dir);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetStrFromSys(const char* path, const char* fileName, char* strValue) {
  char filePath[PATH_MAX];
  sprintf(filePath, "%s/%s", path, fileName);
  int offset = 0;
  FILE* file;
  if ((file = fopen(filePath, "r")) != NULL) {
    while (feof(file) == 0 && ferror(file) == 0 && offset < MAX_STR_LEN) {
      int len = fread(strValue+offset, 1, MAX_STR_LEN-offset, file);
      offset += len;
    }
    fclose(file);
  }
  if (offset == 0) {
    strValue[0] = '\0';
    INFO(NCCL_GRAPH, "Topology detection : could not read %s, ignoring", filePath);
  } else {
    strValue[offset-1] = '\0';
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSetAttrFromSys(struct ncclXmlNode* pciNode, const char* path, const char* fileName, const char* attrName) {
  char strValue[MAX_STR_LEN];
  NCCLCHECK(ncclTopoGetStrFromSys(path, fileName, strValue));
  if (strValue[0] != '\0') { NCCLCHECK(xmlSetAttr(pciNode, attrName, strValue)); }
  TRACE(NCCL_GRAPH, "Read from sys %s/%s -> %s=%s", path, fileName, attrName, strValue);
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromCpu(struct ncclXmlNode* cpuNode, struct ncclXml* xml) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(cpuNode, "affinity", &index));
  if (index == -1) {
    const char* numaId;
    NCCLCHECK(xmlGetAttr(cpuNode, "numaid", &numaId));
    if (numaId == NULL) {
      WARN("GetXmlFromCpu : could not find CPU numa ID.");
      return ncclInternalError;
    }
    // Set affinity
    char cpumaskPath[] = "/sys/devices/system/node/node0000";
    sprintf(cpumaskPath, "/sys/devices/system/node/node%s", numaId);
    NCCLCHECK(ncclTopoSetAttrFromSys(cpuNode, cpumaskPath, "cpumap", "affinity"));
  }

  NCCLCHECK(xmlGetAttrIndex(cpuNode, "arch", &index));
  if (index == -1) {
    // Fill CPU type / vendor / model
#if defined(__PPC__)
    NCCLCHECK(xmlSetAttr(cpuNode, "arch", "ppc64"));
#elif defined(__aarch64__)
    NCCLCHECK(xmlSetAttr(cpuNode, "arch", "arm64"));
#elif defined(__x86_64__)
    NCCLCHECK(xmlSetAttr(cpuNode, "arch", "x86_64"));
#endif
  }

#if defined(__x86_64__)
  NCCLCHECK(xmlGetAttrIndex(cpuNode, "vendor", &index));
  if (index == -1) {
    union {
      struct {
        // CPUID 0 String register order
        uint32_t ebx;
        uint32_t edx;
        uint32_t ecx;
      };
      char vendor[12];
    } cpuid0;

    unsigned unused;
    __cpuid(0, unused, cpuid0.ebx, cpuid0.ecx, cpuid0.edx);
    char vendor[13];
    strncpy(vendor, cpuid0.vendor, 12);
    vendor[12] = '\0';
    NCCLCHECK(xmlSetAttr(cpuNode, "vendor", vendor));
  }

  NCCLCHECK(xmlGetAttrIndex(cpuNode, "familyid", &index));
  if (index == -1) {
    union {
      struct {
        unsigned steppingId:4;
        unsigned modelId:4;
        unsigned familyId:4;
        unsigned processorType:2;
        unsigned resv0:2;
        unsigned extModelId:4;
        unsigned extFamilyId:8;
        unsigned resv1:4;
      };
      uint32_t val;
    } cpuid1;
    unsigned unused;
    __cpuid(1, cpuid1.val, unused, unused, unused);
    int familyId = cpuid1.familyId + (cpuid1.extFamilyId << 4);
    int modelId = cpuid1.modelId + (cpuid1.extModelId << 4);
    NCCLCHECK(xmlSetAttrInt(cpuNode, "familyid", familyId));
    NCCLCHECK(xmlSetAttrInt(cpuNode, "modelid", modelId));
  }
#endif
  return ncclSuccess;
}

ncclResult_t ncclTopoGetPciNode(struct ncclXml* xml, const char* busId, struct ncclXmlNode** pciNode) {
  NCCLCHECK(xmlFindTagKv(xml, "pci", pciNode, "busid", busId));
  if (*pciNode == NULL) {
    NCCLCHECK(xmlAddNode(xml, NULL, "pci", pciNode));
    NCCLCHECK(xmlSetAttr(*pciNode, "busid", busId));
  }
  return ncclSuccess;
}

// Check whether a string is in BDF format or not.
// BDF (Bus-Device-Function) is "BBBB:BB:DD.F" where B, D and F are hex digits.
// There can be trailing chars.
int isHex(char c) { return ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')); }
int checkBDFFormat(char* bdf) {
  if (strlen(bdf) != 12) return 0;
  if ((bdf[4] != ':') || (bdf[7] != ':') || (bdf[10] != '.')) return 0;
  if ((isHex(bdf[0]) == 0) || (isHex(bdf[1]) == 0) || (isHex(bdf[2]) == 0) || (isHex(bdf[3]) == 0) ||
      (isHex(bdf[5]) == 0) || (isHex(bdf[6]) == 0) || (isHex(bdf[8]) == 0) || (isHex(bdf[9]) == 0) ||
      (isHex(bdf[11]) == 0)) return 0;
  return 1;
}

ncclResult_t ncclTopoGetXmlFromSys(struct ncclXmlNode* pciNode, struct ncclXml* xml) {
  // Fill info, then parent
  const char* busId;
  NCCLCHECK(xmlGetAttr(pciNode, "busid", &busId));
  char* path = NULL;
  ncclDebugNoWarn = NCCL_GRAPH;
  if (busId) {
    getPciPath(busId, &path);
  }
  ncclDebugNoWarn = 0;

  if (path) {
    NCCLCHECK(ncclTopoSetAttrFromSys(pciNode, path, "class", "class"));
  }
  int index;
  ncclDebugNoWarn = NCCL_GRAPH;
  NCCLCHECK(xmlGetAttrIndex(pciNode, "vendor", &index));
  if (index == -1) {
    if (path) ncclTopoSetAttrFromSys(pciNode, path, "vendor", "vendor");
  }
  NCCLCHECK(xmlGetAttrIndex(pciNode, "device", &index));
  if (index == -1) {
    if (path) ncclTopoSetAttrFromSys(pciNode, path, "device", "device");
  }
  NCCLCHECK(xmlGetAttrIndex(pciNode, "subsystem_vendor", &index));
  if (index == -1) {
    if (path) ncclTopoSetAttrFromSys(pciNode, path, "subsystem_vendor", "subsystem_vendor");
  }
  NCCLCHECK(xmlGetAttrIndex(pciNode, "subsystem_device", &index));
  if (index == -1) {
    if (path) ncclTopoSetAttrFromSys(pciNode, path, "subsystem_device", "subsystem_device");
  }
  ncclDebugNoWarn = 0;
  NCCLCHECK(xmlGetAttrIndex(pciNode, "link_speed", &index));
  if (index == -1) {
    if (path) {
      char deviceSpeedStr[MAX_STR_LEN];
      float deviceSpeed = FLT_MAX;
      NCCLCHECK(ncclTopoGetStrFromSys(path, "max_link_speed", deviceSpeedStr));
      sscanf(deviceSpeedStr, "%f GT/s", &deviceSpeed);
      char portSpeedStr[MAX_STR_LEN];
      float portSpeed = FLT_MAX;
      NCCLCHECK(ncclTopoGetStrFromSys(path, "../max_link_speed", portSpeedStr));
      if (portSpeedStr[0])
        sscanf(portSpeedStr, "%f GT/s", &portSpeed);
      else
        portSpeed = deviceSpeed;
      NCCLCHECK(xmlSetAttr(pciNode, "link_speed", portSpeed < deviceSpeed ? portSpeedStr : deviceSpeedStr));
    } else {
      NCCLCHECK(xmlSetAttr(pciNode, "link_speed", ""));
    }
  }
  NCCLCHECK(xmlGetAttrIndex(pciNode, "link_width", &index));
  if (index == -1) {
    if (path) {
      char strValue[MAX_STR_LEN];
      NCCLCHECK(ncclTopoGetStrFromSys(path, "max_link_width", strValue));
      int deviceWidth = strtol(strValue, NULL, 0);
      NCCLCHECK(ncclTopoGetStrFromSys(path, "../max_link_width", strValue));
      int portWidth;
      if (strValue[0])
        portWidth = strtol(strValue, NULL, 0);
      else
        portWidth = deviceWidth;
      NCCLCHECK(xmlSetAttrInt(pciNode, "link_width", std::min(deviceWidth,portWidth)));
    } else {
      NCCLCHECK(xmlSetAttr(pciNode, "link_width", ""));
    }
  }

  const char* vendor;
  NCCLCHECK(xmlGetAttr(pciNode, "vendor", &vendor));
  if (vendor != NULL && strcmp(vendor, "0x1000") == 0) { // BCM switch, look for P2P connections
    int nlinks;
    char* peers;
    NCCLCHECK(getBcmLinks(busId, &nlinks, &peers));
    for (int l=0; l<nlinks; l++) {
      char* target = peers+l*BUSID_SIZE;
      struct ncclXmlNode* linkNode;
      NCCLCHECK(xmlGetSubKv(pciNode, "pcilink", &linkNode, "target", target));
      if (linkNode == NULL) {
        NCCLCHECK(xmlAddNode(xml, pciNode, "pcilink", &linkNode));
        NCCLCHECK(xmlSetAttr(linkNode, "target", target));
      }
    }
  }

  struct ncclXmlNode* parent = pciNode->parent;
  if (parent == NULL) {
    if (path) {
      // Save that for later in case next step is a CPU
      char numaIdStr[MAX_STR_LEN];
      NCCLCHECK(ncclTopoGetStrFromSys(path, "numa_node", numaIdStr));
      // Workaround kernel bug for now
      if (strcmp(numaIdStr, "-1") == 0) strcpy(numaIdStr, "0");

      // Go up one level in the PCI tree. Rewind two "/" and follow the upper PCI
      // switch, or stop if we reach a CPU root complex.
      int slashCount = 0;
      int parentOffset;
      for (parentOffset = strlen(path)-1; parentOffset>0; parentOffset--) {
        if (path[parentOffset] == '/') {
          slashCount++;
          path[parentOffset] = '\0';
          int start = parentOffset - 1;
          while (start>0 && path[start] != '/') start--;
          // Check whether the parent path looks like "BBBB:BB:DD.F" or not.
          if (checkBDFFormat(path+start+1) == 0) {
            // This a CPU root complex. Create a CPU tag and stop there.
            struct ncclXmlNode* topNode;
            NCCLCHECK(xmlFindTag(xml, "system", &topNode));
            NCCLCHECK(xmlGetSubKv(topNode, "cpu", &parent, "numaid", numaIdStr));
            if (parent == NULL) {
              NCCLCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
              // NCCLCHECK(xmlSetAttrLong(parent, "host_hash", getHostHash()));
              NCCLCHECK(xmlSetAttr(parent, "numaid", numaIdStr));
            }
          } else if (slashCount == 2) {
            // Continue on the upper PCI switch
            for (int i = strlen(path)-1; i>0; i--) {
              if (path[i] == '/') {
                NCCLCHECK(xmlFindTagKv(xml, "pci", &parent, "busid", path+i+1));
                if (parent == NULL) {
                  NCCLCHECK(xmlAddNode(xml, NULL, "pci", &parent));
                  NCCLCHECK(xmlSetAttr(parent, "busid", path+i+1));
                }
                break;
              }
            }
          }
        }
        if (parent) break;
      }
    } else {
      // No information on /sys, attach GPU to unknown CPU
      NCCLCHECK(xmlFindTagKv(xml, "cpu", &parent, "numaid", "-1"));
      if (parent == NULL) {
        struct ncclXmlNode* topNode;
        NCCLCHECK(xmlFindTag(xml, "system", &topNode));
        NCCLCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
        // NCCLCHECK(xmlSetAttrLong(parent, "host_hash", getHostHash()));
        NCCLCHECK(xmlSetAttr(parent, "numaid", "-1"));
        NCCLCHECK(ncclTopoGetXmlFromCpu(parent, xml));
      }
    }
    pciNode->parent = parent;
    // Keep PCI sub devices ordered by PCI Bus ID (Issue #820)
    int subIndex = parent->nSubs;
    const char* newBusId;
    NCCLCHECK(xmlGetAttrStr(pciNode, "busid", &newBusId));
    for (int s=0; s<parent->nSubs; s++) {
      const char* busId;
      NCCLCHECK(xmlGetAttr(parent->subs[s], "busid", &busId));
      if (busId != NULL && strcmp(newBusId, busId) < 0) { subIndex = s; break; }
    }
    if (parent->nSubs == MAX_SUBS) {
      WARN("Error : XML parser is limited to %d subnodes", MAX_SUBS);
      return ncclInternalError;
    }
    for (int s = parent->nSubs; s > subIndex; s--) parent->subs[s] = parent->subs[s-1];
    parent->subs[subIndex] = pciNode;
    parent->nSubs++;
  }
  if (strcmp(parent->name, "pci") == 0) {
    NCCLCHECK(ncclTopoGetXmlFromSys(parent, xml));
  } else if (strcmp(parent->name, "cpu") == 0) {
    NCCLCHECK(ncclTopoGetXmlFromCpu(parent, xml));
  }
  free(path);
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromGpu(struct ncclXmlNode* pciNode, uint32_t rocmDev, struct ncclXml* xml, struct ncclXmlNode** gpuNodeRet) {
  struct ncclXmlNode* gpuNode = NULL;
  NCCLCHECK(xmlGetSub(pciNode, "gpu", &gpuNode));
  if (gpuNode == NULL) NCCLCHECK(xmlAddNode(xml, pciNode, "gpu", &gpuNode));

  int index = -1;

  int dev = -1;
  NCCLCHECK(xmlGetAttrIndex(gpuNode, "dev", &index));
  if (index == -1) {
    NCCLCHECK(xmlSetAttrInt(gpuNode, "dev", rocmDev));
  }
  NCCLCHECK(xmlGetAttrInt(gpuNode, "dev", &dev));
  if (dev == -1) { *gpuNodeRet = NULL; return ncclSuccess; }

  NCCLCHECK(xmlGetAttrIndex(gpuNode, "sm", &index));
  if (index == -1) {
    cudaDeviceProp devProp;
    CUDACHECK(cudaGetDeviceProperties(&devProp, 0));
    NCCLCHECK(xmlSetAttrInt(gpuNode, "sm", devProp.multiProcessorCount));
  }
  int sm;
  NCCLCHECK(xmlGetAttrInt(gpuNode, "sm", &sm));

  const char* gcn;
  const char* gcnArchName;
  NCCLCHECK(xmlGetAttrIndex(gpuNode, "gcn", &index));
  if (index == -1) {
    hipDeviceProp_t devProp;
    CUDACHECK(hipGetDeviceProperties(&devProp, 0));
    //extract only the releveant info from the gcnArchName attribute
    //e.g.: convert "gfx908:sramecc+:xnack-" to "gfx908"
    char gcnArchNameSubstr[128];
    GcnArchNameFormat(devProp.gcnArchName, gcnArchNameSubstr);
    gcn = gcnArchNameSubstr;
    NCCLCHECK(xmlSetAttr(gpuNode, "gcn", gcn));
  }
  NCCLCHECK(xmlGetAttr(gpuNode, "gcn", &gcn));
  convertGcnArchToGcnArchName(gcn, &gcnArchName);
  if (gcn != gcnArchName) {
     NCCLCHECK(xmlSetAttr(gpuNode, "gcn", gcnArchName));
  }

  rcclHipDeviceArch_t arch;
  NCCLCHECK(xmlGetAttrIndex(gpuNode, "arch", &index));
  if (index == -1) {
    hipDeviceProp_t devProp;
    CUDACHECK(hipGetDeviceProperties(&devProp, 0));
    memcpy(&arch.arch, &devProp.arch, sizeof(hipDeviceArch_t));
    NCCLCHECK(xmlSetAttrInt(gpuNode, "arch", arch.value));
  }
  NCCLCHECK(xmlGetAttrInt(gpuNode, "arch", &arch.value));

  struct ncclXmlNode* nvlNode = NULL;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  NCCLCHECK(xmlGetSub(gpuNode, "xgmi", &nvlNode));
#else
  NCCLCHECK(xmlGetSub(gpuNode, "nvlink", &nvlNode));
#endif
  if (nvlNode == NULL) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#ifdef RSMI_SUPPORT_XHCL_TOPO
    uint32_t linkNum = 0;
    uint32_t linkState = 0;
    uint64_t pciBdf = 0;

    NCCLCHECK(rocm_smi_dev_xhcl_link_number_get(rocmDev, &linkNum));
    for (int j = 0; j < linkNum; j++) {
      bool isSkip = false;
      NCCLCHECK(rocm_smi_dev_xhcl_link_state_get(rocmDev, j, &linkState));
      if (linkState == 0) {
        continue;
      }

      NCCLCHECK(rocm_smi_dev_xhcl_link_remote_bdfid_get(rocmDev, j, &pciBdf, &isSkip));
      if (isSkip) {
        continue;
      }
      char busIdStr[] = "00000000:00:00.0";
      snprintf(busIdStr, sizeof(busIdStr), "%04lx:%02lx:%02lx.%01lx", (pciBdf) >> 32, (pciBdf & 0xff00) >> 8, (pciBdf & 0xf8) >> 3, (pciBdf & 0x7));
      char lowerId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      for (int c=0; c<NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE; c++) {
        lowerId[c] = tolower(busIdStr[c]);
        if (busIdStr[c] == 0) break;
      }
      NCCLCHECK(xmlGetSubKv(gpuNode, "xgmi", &nvlNode, "target", lowerId));
      if (nvlNode == NULL) {
        NCCLCHECK(xmlAddNode(xml, gpuNode, "xgmi", &nvlNode));
        NCCLCHECK(xmlSetAttr(nvlNode, "target", lowerId));
        NCCLCHECK(xmlSetAttrInt(nvlNode, "count", 1));
      } else {
        int count;
        NCCLCHECK(xmlGetAttrInt(nvlNode, "count", &count));
        NCCLCHECK(xmlSetAttrInt(nvlNode, "count", count+1));
      }
    }
#else
    const char* busId;
    NCCLCHECK(xmlGetAttr(pciNode, "busid", &busId));
    uint32_t deviceCnt;
    NCCLCHECK(rocm_smi_getNumDevice(&deviceCnt));
    for (int i=0; i<deviceCnt; i++) {
      if (i != dev) {
        RSMI_IO_LINK_TYPE rsmi_type;
        int hops, count;
        if (rocm_smi_getLinkInfo(dev, i, &rsmi_type, &hops, &count) == ncclSuccess) {
          if (rsmi_type >= RSMI_IOLINK_TYPE_XGMI && hops >= 1) {
            char busIdStr[] = "00000000:00:00.0";
            NCCLCHECK(rocm_smi_getDevicePciBusIdString(i, busIdStr, sizeof(busIdStr)));
            char lowerId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
            for (int c=0; c<NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE; c++) {
              lowerId[c] = tolower(busIdStr[c]);
              if (busIdStr[c] == 0) break;
            }
            NCCLCHECK(xmlGetSubKv(gpuNode, "xgmi", &nvlNode, "target", lowerId));
            if (nvlNode == NULL) {
              NCCLCHECK(xmlAddNode(xml, gpuNode, "xgmi", &nvlNode));
              NCCLCHECK(xmlSetAttr(nvlNode, "target", lowerId));
              NCCLCHECK(xmlSetAttrInt(nvlNode, "count", count));
            }
          }
        }
      }
    }
#endif
#else
    // NVML NVLink detection
    int maxNvLinks = (sm < 60) ? 0 : (sm < 70) ? 4 : (sm < 80) ? 6 : (sm < 90) ? 12 : 18;

    if (maxNvLinks > 0 && nvmlDev == NULL) {
      WARN("No NVML device handle. Skipping nvlink detection.");
      maxNvLinks = 0;
    }

    for (int l=0; l<maxNvLinks; ++l) {
      // Check whether we can use this NVLink for P2P
      unsigned canP2P;
      if ((ncclNvmlDeviceGetNvLinkCapability(nvmlDev, l, NVML_NVLINK_CAP_P2P_SUPPORTED, &canP2P) != ncclSuccess) || !canP2P) continue;

      // Make sure the Nvlink is up. The previous call should have trained the link.
      nvmlEnableState_t isActive = NVML_FEATURE_DISABLED;
#if CUDART_VERSION >= 11080
      if (sm >= 90) {
        nvmlFieldValue_t fv;
        fv.fieldId = NVML_FI_DEV_NVLINK_GET_STATE;
        fv.scopeId = l;
        // fv.value will contain NV_FEATURE_ENABLED or NV_FEATURE_DISABLED
        if ((ncclNvmlDeviceGetFieldValues(nvmlDev, 1, &fv) == ncclSuccess) && (fv.nvmlReturn == NVML_SUCCESS))
          isActive = (nvmlEnableState_t) fv.value.uiVal;
      } else /* FALLTHRU to GetNvLinkState if before SM90 */
#endif
      {
        (void) ncclNvmlDeviceGetNvLinkState(nvmlDev, l, &isActive);
      }
      if (isActive != NVML_FEATURE_ENABLED) continue;

      // Try to figure out what's on the other side of the NVLink
      nvmlPciInfo_t remoteProc;
      if (ncclNvmlDeviceGetNvLinkRemotePciInfo(nvmlDev, l, &remoteProc) != ncclSuccess) continue;

      // Make a lower case copy of the bus ID for calling ncclDeviceType
      // PCI system path is in lower case
      char* p = remoteProc.busId;
      char lowerId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      for (int c=0; c<NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE; c++) {
        lowerId[c] = tolower(p[c]);
        if (p[c] == 0) break;
      }

      NCCLCHECK(xmlGetSubKv(gpuNode, "nvlink", &nvlNode, "target", lowerId));
      if (nvlNode == NULL) {
        NCCLCHECK(xmlAddNode(xml, gpuNode, "nvlink", &nvlNode));
        NCCLCHECK(xmlSetAttr(nvlNode, "target", lowerId));
        NCCLCHECK(xmlSetAttrInt(nvlNode, "count", 1));
      } else {
        int count;
        NCCLCHECK(xmlGetAttrInt(nvlNode, "count", &count));
        NCCLCHECK(xmlSetAttrInt(nvlNode, "count", count+1));
      }
    }
#endif
  }
#if CUDART_VERSION >= 11080
  struct ncclXmlNode* c2cNode = NULL;
  NCCLCHECK(xmlGetSub(gpuNode, "c2c", &c2cNode));
  if (c2cNode == NULL) {
      if (sm >= 90) {
        int c2cLinksCount = 0;
        nvmlFieldValue_t fv;
        fv.fieldId = NVML_FI_DEV_C2C_LINK_COUNT;
        if ((ncclNvmlDeviceGetFieldValues(nvmlDev, 1, &fv) == ncclSuccess) && (fv.nvmlReturn == NVML_SUCCESS)) {
          c2cLinksCount = fv.value.uiVal;
          int bw = 0;
	  int count = 0;
          for (int l=0; l<c2cLinksCount; l++) {
            nvmlFieldValue_t fvs[2];
            fvs[0].fieldId = NVML_FI_DEV_C2C_LINK_GET_STATUS;
            fvs[0].scopeId = l;
            fvs[1].fieldId = NVML_FI_DEV_C2C_LINK_GET_MAX_BW;
            fvs[1].scopeId = l;
            if ((ncclNvmlDeviceGetFieldValues(nvmlDev, 2, fvs) == ncclSuccess) &&
                (fvs[0].nvmlReturn == NVML_SUCCESS) &&
                (fvs[0].value.uiVal == 1) &&
                (fvs[1].nvmlReturn == NVML_SUCCESS)) {
              bw = fvs[1].value.uiVal;
	      count++;
            }
          }
          if (count > 0) {
            NCCLCHECK(xmlAddNode(xml, gpuNode, "c2c", &c2cNode));
            NCCLCHECK(xmlSetAttrInt(c2cNode, "bw", bw));
            NCCLCHECK(xmlSetAttrInt(c2cNode, "count", count));
          }
        }
      }
  }
#endif
  // Fill target classes
  for (int s=0; s<gpuNode->nSubs; s++) {
    struct ncclXmlNode* sub = gpuNode->subs[s];
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    if (strcmp(sub->name, "xgmi") != 0) continue;
#else
    if (strcmp(sub->name, "nvlink") != 0) continue;
#endif
    int index;
    NCCLCHECK(xmlGetAttrIndex(sub, "tclass", &index));
    if (index == -1) {
      const char* busId;
      NCCLCHECK(xmlGetAttr(sub, "target", &busId));
      char* path;
      ncclDebugNoWarn = NCCL_GRAPH;
      getPciPath(busId, &path);
      ncclDebugNoWarn = 0;
      if (path == NULL || strcmp(busId, "fffffff:ffff:ff") == 0) {
        // Remote NVLink device is not visible inside this VM. Assume NVSwitch.
        NCCLCHECK(xmlSetAttr(sub, "tclass", "0x068000"));
      } else {
        NCCLCHECK(ncclTopoSetAttrFromSys(sub, path, "class", "tclass"));
        free(path);
      }
    }
  }
  *gpuNodeRet = gpuNode;
  return ncclSuccess;
}

ncclResult_t ncclTopoFillGpu(struct ncclXml* xml, const char* busId, struct ncclXmlNode** gpuNode) {
  struct ncclXmlNode* node;
  NCCLCHECK(ncclTopoGetPciNode(xml, busId, &node));
  NCCLCHECK(xmlSetAttrIfUnset(node, "class", "0x03"));
  NCCLCHECK(ncclTopoGetXmlFromSys(node, xml));
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  uint32_t devIndex = 0;
  static int rocmsmiInit = 0;
  if (rocmsmiInit == 0) {
    NCCLCHECK(rocm_smi_init());
  }
  NCCLCHECK(rocm_smi_getDeviceIndexByPciBusId(busId, &devIndex));
  NCCLCHECK(ncclTopoGetXmlFromGpu(node, devIndex, xml, gpuNode));
#else
  nvmlDevice_t nvmlDev;
  NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));
  NCCLCHECK(ncclTopoGetXmlFromGpu(node, nvmlDev, xml, gpuNode));
#endif
  return ncclSuccess;
}

// Returns the subsystem name of a path, i.e. the end of the path
// where sysPath/subsystem points to.
ncclResult_t ncclTopoGetSubsystem(const char* sysPath, char* subSys) {
  char subSysPath[PATH_MAX];
  sprintf(subSysPath, "%s/subsystem", sysPath);
  char* path = realpath(subSysPath, NULL);
  if (path == NULL) {
    subSys[0] = '\0';
  } else {
    int offset;
    for (offset = strlen(path); offset > 0 && path[offset] != '/'; offset--);
    strcpy(subSys, path+offset+1);
    free(path);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoFillNet(struct ncclXml* xml, const char* pciPath, const char* netName, struct ncclXmlNode** netNode) {
  NCCLCHECK(xmlFindTagKv(xml, "net", netNode, "name", netName));
  //if (*netNode != NULL) return ncclSuccess;

  const char* pciSysPath = pciPath;
  if (pciSysPath) {
    char subSystem[PATH_MAX];
    NCCLCHECK(ncclTopoGetSubsystem(pciSysPath, subSystem));
    // This is not a PCI device (virtual, usb, ...).
    if (strcmp(subSystem, "pci") != 0) {
      INFO(NCCL_GRAPH, "Topology detection: network path %s is not a PCI device (%s). Attaching to first CPU", pciSysPath, subSystem);
      pciSysPath = NULL;
    }
  }

  struct ncclXmlNode* parent = NULL;
  if (pciSysPath) {
    int offset;
    for (offset=strlen(pciSysPath)-1; pciSysPath[offset] != '/'; offset--);
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    strcpy(busId, pciSysPath+offset+1);
    NCCLCHECK(ncclTopoGetPciNode(xml, busId, &parent));
    NCCLCHECK(xmlSetAttrIfUnset(parent, "class", "0x02"));
    NCCLCHECK(ncclTopoGetXmlFromSys(parent, xml));
  } else {
    // Virtual NIC, no PCI device, attach to NUMA 0
    struct ncclXmlNode* topNode;
    NCCLCHECK(xmlFindTag(xml, "system", &topNode));
    NCCLCHECK(xmlGetSubKv(topNode, "cpu", &parent, "numaid", "0"));
    if (parent == NULL) {
      NCCLCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
      NCCLCHECK(xmlSetAttr(parent, "numaid", "0"));
      NCCLCHECK(ncclTopoGetXmlFromCpu(parent, xml));
    }
  }

  struct ncclXmlNode* nicNode = NULL;
  NCCLCHECK(xmlGetSub(parent, "nic", &nicNode));
  if (nicNode == NULL) {
    NCCLCHECK(xmlAddNode(xml, parent, "nic", &nicNode));
  }

  
  if (*netNode == NULL) {
    NCCLCHECK(xmlAddNode(xml, nicNode, "net", netNode));
    NCCLCHECK(xmlSetAttr(*netNode, "name", netName));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoTrimXmlRec(struct ncclXmlNode* node, int* keep) {
  const char* str;
  NCCLCHECK(xmlGetAttr(node, "keep", &str));
  if (str && strcmp(str, "1") == 0) {
    NCCLCHECK(xmlUnsetAttr(node, "keep"));
    *keep = 1;
  } else {
    // Copy nSubs and subs as they could change as we trim recursively.
    struct ncclXmlNode* subs[MAX_SUBS];
    int nSubs = node->nSubs;
    memcpy(subs, node->subs, node->nSubs*sizeof(struct ncclXmlNode*));
    *keep = 0;
    for (int s=0; s<nSubs; s++) {
      int k = 0;
      NCCLCHECK(ncclTopoTrimXmlRec(subs[s], &k));
      *keep += k;
    }
    if (*keep == 0 && // Trim PCI switches or CPU with no used GPU/NIC under them.
        (strcmp(node->name, "pci") == 0 || strcmp(node->name, "cpu") == 0 || strcmp(node->name, "net") == 0)) {
      NCCLCHECK(xmlRemoveNode(node));
    }
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoTrimXml(struct ncclXml* xml) {
  int keep = 0;
  NCCLCHECK(ncclTopoTrimXmlRec(xml->nodes, &keep));
  return ncclSuccess;
}

/**************************************************/
/* Parser rules for the user-defined graph search */
/**************************************************/

ncclResult_t ncclTopoXmlGraphLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadChannel(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "net", ncclTopoXmlGraphLoadNet }, { "gpu", ncclTopoXmlGraphLoadGpu } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraph(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "channel", ncclTopoXmlGraphLoadChannel } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraphs(FILE* file, struct ncclXml* xmlGraph, struct ncclXmlNode* head) {
  int version;
  NCCLCHECK(xmlGetAttrInt(head, "version", &version));
  if (version != NCCL_GRAPH_XML_VERSION) {
    WARN("XML Graph has wrong version %d, %d needed", version, NCCL_GRAPH_XML_VERSION);
    return ncclInvalidUsage;
  }
  const char* name;
  NCCLCHECK(xmlGetAttr(head, "name", &name));
  if (name != NULL) INFO(NCCL_GRAPH, "Loading graphs for topology %s", name);
  else INFO(NCCL_GRAPH, "Loading graphs");

  struct xmlHandler handlers[] = { { "graph", ncclTopoXmlGraphLoadGraph } };
  NCCLCHECK(xmlLoadSub(file, xmlGraph, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlGraphFromFile(const char* xmlGraphFile, struct ncclXml* xml) {
  FILE* file = fopen(xmlGraphFile, "r");
  if (file == NULL) {
    WARN("Could not open XML graph file %s : %s", xmlGraphFile, strerror(errno));
    return ncclSystemError;
  }
  struct xmlHandler handlers[] = { { "graphs", ncclTopoXmlGraphLoadGraphs } };
  xml->maxIndex = 0;
  NCCLCHECK(xmlLoadSub(file, xml, NULL, handlers, 1));
  fclose(file);
  return ncclSuccess;
}

NCCL_PARAM(BasicTopoDumpFileRank, "TOPO_DUMP_FILE_RANK", 0);

/*
 * 函数功能：基础拓扑生成功能入口函数，内部逻辑如下：
            1.从NCCL_TOPO_MAPPING_FILE或ROCM_PATH获取映射文件路径；
            2.收集全部pcieSwitch、nic、gpu节点的busid，保存到全局的g_busIdSet中
            3.收集环境信息，用于后续默认映射文件读取的组名辨别
            4.解析文件，将xml内容解析到ncclXml结构体对象中；
              a.解析xml文件；
              b.遇到gpu和nic节点则填充缺失的pci层级;
              c.填充pci层级缺失的busId属性;
            5.移除ncclXml结构体对象中name为group的节点;
 * 参数说明：
 *   comm: 输入参数，通讯组指针
 *   xml: 输出参数，XML解析上下文结构体
 * 返回值：ncclSuccess表示成功，其他错误码表示解析/处理失败
 */
ncclResult_t ncclTopoGetXmlFromMappingFile(struct ncclComm* comm, struct ncclXml* xml) {
  char filePath[PATH_MAX] = {};
  const char* xmlMappingFile = ncclGetEnv("NCCL_TOPO_MAPPING_FILE");
  if (xmlMappingFile) {
    isTopoMappingDefaultFile = false;
    snprintf(filePath, sizeof(filePath),"%s", xmlMappingFile);
  } else {
    char* rocmPath = getenv("ROCM_PATH");
    if (rocmPath != NULL) {
      snprintf(filePath, sizeof(filePath),"%s%s", rocmPath, "/rccl/lib/topo_mapping_default.xml");
      isTopoMappingDefaultFile = true;
    } else {
      INFO(NCCL_ENV, "No topo mapping file has been specified.");
      return ncclSuccess;
    }
  }

  //Collect all devices busids
  static std::once_flag init_flag;
  std::call_once(init_flag,[&comm](){ NCCLCHECK(collectNodesBusIds(comm)); });

  // Make sure we have at least one IB NIC
  if (g_busIdSet.nicBusId.empty()) {
    WARN("No IB NIC found, the generation of the basic topology has been terminated.");
    return ncclSuccess;
  }

  //Collect environmental information
  std::string envTypeInfoStr = envTypeInfo::getEnvTypeInfoStr();
  if (envTypeInfoStr.empty()) {
    WARN("Unable to complete the environmental inspection, the generation of the basic topology has been terminated.");
    return ncclSuccess;
  }
  INFO(NCCL_GRAPH, "After environmental exploration, the environmental key word is %s", envTypeInfoStr.c_str());
  
  //Parse the XML file and fill in the "busid" attribute
  FILE* file = fopen(filePath, "r");
  if (file == NULL) {
    WARN("Could not open XML topology mapping file %s : %s", filePath, strerror(errno));
    return ncclSuccess;
  }
  INFO(NCCL_GRAPH, "Loading topology mapping file %s", filePath);
  struct xmlHandler handlers[] = { { "system", ncclTopoXmlGraphLoadMappingSystem } };
  xml->maxIndex = 0;
  NCCLCHECK(xmlLoadSub(file, xml, NULL, handlers, 1));
  fclose(file);

  // remove group\slot node if exists
  NCCLCHECK(ncclxmlRemoveAllNodeByName(xml,"group"));
  NCCLCHECK(ncclxmlRemoveAllNodeByName(xml,"slot"));

  const char* xmlTopoFile = ncclGetEnv("NCCL_BASIC_TOPO_DUMP_FILE");
  if (xmlTopoFile && comm->rank == ncclParamBasicTopoDumpFileRank()) {
    INFO(NCCL_ENV, "NCCL_BASIC_TOPO_DUMP_FILE set by environment to %s", xmlTopoFile);
    NCCLCHECK(ncclTopoDumpXmlToFile(xmlTopoFile, xml));
  }
  return ncclSuccess;
}

/*
 * 函数功能：删除全部节点名为nodeName的节点
 * 参数说明：
 *   xml: 输出参数，XML解析上下文结构体，节点集合；
 *   nodeName：输入参数，节点名
 * 返回值：ncclSuccess表示成功，其他错误码表示解析/处理失败
 */
ncclResult_t ncclxmlRemoveAllNodeByName(struct ncclXml* xml,const char* nodeName) {
  if (!xml || !nodeName) {
    return ncclSuccess;
  }

  struct ncclXmlNode* node;
  NCCLCHECK(xmlFindTag(xml,nodeName,&node));
  while(node) {
    struct ncclXmlNode* parentNode = node->parent;
    if (parentNode) {
      for(int i = 0; i < node->nSubs; i++) {
        if (parentNode->nSubs == MAX_SUBS) {
            WARN("Error : XML parser is limited to %d subnodes", MAX_SUBS);
            return ncclInternalError;
        }
        parentNode->subs[parentNode->nSubs++] = node->subs[i];
        node->subs[i]->parent = parentNode;
      }
    }
    NCCLCHECK(xmlRemoveNode(node));
    memset(node, 0, sizeof(struct ncclXmlNode));
    NCCLCHECK(xmlFindTag(xml,nodeName,&node));
  }
  return ncclSuccess;
}

/*
 * 函数功能：解析最外层system层，使用ncclTopoXmlGraphLoadGroup处理group标签
 * 参数说明：
 *   file: 输入参数，配置文件
 *   xml: 输出参数，XML解析上下文结构体
 *   head：父节点
 * 返回值：ncclSuccess表示成功，其他错误码表示解析/处理失败
 */
ncclResult_t ncclTopoXmlGraphLoadMappingSystem(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  int version;
  NCCLCHECK(xmlGetAttrInt(head, "version", &version));
  if (version != NCCL_TOPO_XML_VERSION) {
    WARN("XML Topology mapping has wrong version %d, %d needed", version, NCCL_TOPO_XML_VERSION);
    return ncclInvalidUsage;
  }

  struct xmlHandler handlers[] = { { "group", ncclTopoXmlGraphLoadMappingGroup } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}


bool containsKeyWord(const std::string& groupName, const std::string& keyWord) {
    std::size_t start = 0;
    while (true) {
        std::size_t end = groupName.find('|', start);
        std::string token = groupName.substr(start, end - start);
        if (token == keyWord) return true;
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return false;
}
/*
 * 函数功能：解析group层，使用ncclTopoXmlGraphLoadMappingCpu处理cpu标签，额外的逻辑如下：
            1.当使用默认映射配置文件时，尝试从envTypeInfo中获取当前环境标签，若能获取则以环境标签为group name；
            2.当使用自定义映射配置文件时，解析第一个合格的group；
            注：group标签要求必须有name属性
 * 参数说明：
 *   file: 输入参数，配置文件
 *   xml: 输出参数，XML解析上下文结构体
 *   head：父节点
 * 返回值：ncclSuccess表示成功，其他错误码表示解析/处理失败
 */
ncclResult_t ncclTopoXmlGraphLoadMappingGroup(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  const char* name = NULL;
  std::string xmlMappingGroupName="";
  thread_local bool isGotGroup = false;
  
  // 始终只解析一个组
  if (isGotGroup) {
    NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
    return ncclSuccess;
  }

  NCCLCHECK(xmlGetAttr(head, "name", &name));
  if (name != NULL) {
    if (isTopoMappingDefaultFile) {
      xmlMappingGroupName = envTypeInfo::getEnvTypeInfoStr();
    }

    if (!isTopoMappingDefaultFile || containsKeyWord(name, xmlMappingGroupName)) {
      INFO(NCCL_GRAPH, "parseing topology mapping group: %s", name);
      struct xmlHandler handlers[] = { { "cpu", ncclTopoXmlGraphLoadMappingCpu } };
      NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
      isGotGroup = true;
    } else {
      INFO(NCCL_GRAPH, "skip topology mapping group: %s ,expected group name is %s", name,xmlMappingGroupName.c_str());
      NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
    }
  } else {
    INFO(NCCL_GRAPH, "Ignoring topology unnamed mapping group:");
    NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  }

  return ncclSuccess;
}

/*
 * 函数功能：解析cpu层，使用ncclTopoXmlGraphLoadMappingPci处理pci标签
 * 参数说明：
 *   file: 输入参数，配置文件
 *   xml: 输出参数，XML解析上下文结构体
 *   head：父节点
 * 返回值：ncclSuccess表示成功，其他错误码表示解析/处理失败
 */
ncclResult_t ncclTopoXmlGraphLoadMappingCpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "pci", ncclTopoXmlGraphLoadMappingPci } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

/*
 * 函数功能：解析pci层：
          （1）当判断为pcieSwtich节点时，从全局变量g_busIdSet中获取并设置当前pci节点busid属性
          （2）传递gpu、nic和slot标签处理函数
 * 参数说明：
 *   file: 输入参数，配置文件
 *   xml: 输出参数，XML解析上下文结构体
 *   head：父节点
 * 返回值：ncclSuccess表示成功，其他错误码表示解析/处理失败
 */
ncclResult_t ncclTopoXmlGraphLoadMappingPci(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  thread_local int pcieSwitchCount = 0;
  //set busid for pcieSwtich
  if (head->parent && strcmp(head->parent->name,"cpu") == 0) {
    if (pcieSwitchCount >= g_busIdSet.pcieSwitchBusId.size()) {
      WARN("Error : The number of PCIe Switch labels [%d] exceeds the number of collected PCIe Switch busids [%lu], ignore the excess PCIe Switch labels."
        ,pcieSwitchCount+1,g_busIdSet.pcieSwitchBusId.size());
      NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
      NCCLCHECK(xmlRemoveNode(head));
      memset(head, 0, sizeof(struct ncclXmlNode));
      return ncclSuccess;
    }
    NCCLCHECK(xmlSetAttr(head, "busid",g_busIdSet.pcieSwitchBusId[pcieSwitchCount++].c_str()));
  }

  struct xmlHandler handlers[] = { { "gpu", ncclTopoXmlGraphLoadMappingGpu }, { "nic", ncclTopoXmlGraphLoadMappingNic}, { "slot", ncclTopoXmlGraphLoadMappingSlot}  };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 3));
  return ncclSuccess;
}

/*
 * 函数功能：解析gpu层：
          （1）为gpu节点填充两个pci父节点(新增两个节点，将其作为pcie(inner)和gpu节点，原gpu节点转换为pcie(middle)节点，完善层级结构)
          （2）按照gpu节点的dev属性拼接key，从全局变量g_busIdSet中查找busid并设置两个pci父节点的busid属性
 * 参数说明：
 *   file: 输入参数，配置文件
 *   xml: 输出参数，XML解析上下文结构体
 *   head：父节点
 * 返回值：ncclSuccess表示成功，其他错误码表示解析/处理失败
 */
ncclResult_t ncclTopoXmlGraphLoadMappingGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  // before: pcie(top)->gpu
  // after: pcie(top)->pcie(middle)->pcie(inner)->gpu

  // pcie:middle
  struct ncclXmlNode* middlePciNode = head;

  // pcie:inner
  struct ncclXmlNode* innerPciNode;
  // create pci node
  NCCLCHECK(XmlAddAndCopyNode(xml, middlePciNode, "pci", &innerPciNode));
  innerPciNode->type = NODE_TYPE_CLOSE;

  // gpu
  struct ncclXmlNode* gpuNode;
  // create gpu node and copy attrs from middlePciNode(head)
  NCCLCHECK(XmlAddAndCopyNode(xml, innerPciNode, "gpu", &gpuNode, middlePciNode, true));
  gpuNode->type = NODE_TYPE_SINGLE;

  // reset pcie:middle
  strncpy(middlePciNode->name, "pci", sizeof(middlePciNode->name)-1);
  memset(middlePciNode->attrs, 0, sizeof(middlePciNode->attrs));
  middlePciNode->nAttrs = 0;
  middlePciNode->type = NODE_TYPE_CLOSE;

  // set busid for pcie(middle)、pcie(inner)
  int dev;
  NCCLCHECK(xmlGetAttrIntDefault(gpuNode, "dev", &dev, -1));
  if (-1 != dev) { 
    char gpuKey[MAX_STR_LEN] = {};
    snprintf(gpuKey, sizeof(gpuKey),"%s%d", MAPPING_GPU_KEY_PREFIX, dev);
    // only set two gpus busid
    if (g_busIdSet.gpuBusId.size() >= 2 && g_busIdSet.gpuBusId[1].find(gpuKey) != g_busIdSet.gpuBusId[1].end() && g_busIdSet.gpuBusId[0].find(gpuKey) != g_busIdSet.gpuBusId[0].end()) {
      NCCLCHECK(xmlSetAttr(innerPciNode, "busid", g_busIdSet.gpuBusId[0][gpuKey].c_str()));
      NCCLCHECK(xmlSetAttr(middlePciNode, "busid", g_busIdSet.gpuBusId[1][gpuKey].c_str()));
      // remove dev Attr
      NCCLCHECK(xmlUnsetAttr(gpuNode, "dev"));
    } else {
      WARN("Error in the busID of the GPU or the current GPU[%d] cannot be found.",dev);
    }
  } else {
    WARN("gpu dev attr no exit or out of range.");
  }

  //set link_speed\link_width for pcie(middle)、pcie(inner) if exists
  char attrs[][MAX_STR_LEN] = {"link_speed","link_width"};
  for (int i=0; i < (sizeof(attrs)/sizeof(attrs[0])); i++) {
    NCCLCHECK(xmlCopyAttr(innerPciNode,gpuNode,attrs[i]));
    NCCLCHECK(xmlCopyAttr(middlePciNode,gpuNode,attrs[i]));
    NCCLCHECK(xmlUnsetAttr(gpuNode, attrs[i]));
  }
  
  NCCLCHECK(xmlLoadSub(file, xml, gpuNode, NULL, 0));
  return ncclSuccess;
}

/*
 * 函数功能：解析nic层，当多个网卡的busid相同时，需要在一个nic下生成多个net标签，对此将分类讨论：
            1.按当前nic标签的网卡名查找busid：
            2.若没有当前busid的pci标签时，进行结构填充：
              （1）为nic节点填充一个pci父节点(新增一个nic节点，将原nic节点内容复制到新节点后，原nic节点改为pci节点)
              （2）为nic节点填充一个net子节点；
              （3）属性设置：设置pci父节点的busid属性、若存在则为pci父节点填充link_speed\link_width属性，若存在则为net子节点填充speed属性；
            3.若当前busid的pci标签已存在。则仅新增net子节点，并尝试填充net子节点的speed属性；
 * 参数说明：
 *   file: 输入参数，配置文件
 *   xml: 输出参数，XML解析上下文结构体
 *   head：父节点
 * 返回值：ncclSuccess表示成功，其他错误码表示解析/处理失败
 */
ncclResult_t ncclTopoXmlGraphLoadMappingNic(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  std::string busId;
  std::string nicNameStr;
  const char* nicName = NULL;
  NCCLCHECK(xmlGetAttr(head, "id", &nicName));
  if (NULL != nicName && g_busIdSet.nicBusId.find(nicName) != g_busIdSet.nicBusId.end()) {
    nicNameStr = nicName;
    busId = g_busIdSet.nicBusId[nicName];
    // remove id Attr
    NCCLCHECK(xmlUnsetAttr(head, "id"));
  } else {
    WARN("Nic dev attr no exit or the current NIC[%s] cannot be found.",nicName!=NULL?nicName:"NULL");
    NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
    NCCLCHECK(xmlRemoveNode(head));
    memset(head, 0, sizeof(struct ncclXmlNode));
    return ncclSuccess;
  }

  // find if pci node exists
  struct ncclXmlNode* innerPciNode;
  struct ncclXmlNode* nicNode;
  struct ncclXmlNode* netNode;
  NCCLCHECK(xmlFindTagKv(xml, "pci", &innerPciNode, "busid", busId.c_str()));
  if (NULL == innerPciNode) {
    // fill topo nodes
      // before: pcie(top)->nic
      // after: pcie(top)->pcie(inner)->nic->net

    // pcie:inner
    struct ncclXmlNode* innerPciNode = head;
    
    // nic
    NCCLCHECK(XmlAddAndCopyNode(xml, innerPciNode, "nic", &nicNode, head, true));
    nicNode->type = NODE_TYPE_CLOSE;
   
    // reset pcie:inner
    strncpy(innerPciNode->name, "pci", sizeof(innerPciNode->name)-1);
    memset(innerPciNode->attrs, 0, sizeof(innerPciNode->attrs));
    innerPciNode->nAttrs = 0;
    innerPciNode->type = NODE_TYPE_CLOSE;
    NCCLCHECK(xmlSetAttr(innerPciNode, "busid", busId.c_str()));

    //set link_speed\link_width for pcie(inner) if exists
    char attrs[][MAX_STR_LEN] = {"link_speed","link_width"};
    for (int i=0; i < (sizeof(attrs)/sizeof(attrs[0])); i++) {
      NCCLCHECK(xmlCopyAttr(innerPciNode,nicNode,attrs[i]));
      NCCLCHECK(xmlUnsetAttr(nicNode, attrs[i]));
    }

    // add net node
    NCCLCHECK(ncclTopoMappingGetNetNode(xml,nicNode,nicNameStr.c_str(),&netNode));
    // copy speed attr from nic node if exists
    NCCLCHECK(xmlCopyAttr(netNode,nicNode,"speed"));
    NCCLCHECK(xmlUnsetAttr(nicNode, "speed"));
  } else {
    // find nic node
    NCCLCHECK(xmlGetSub(innerPciNode,"nic",&nicNode));
    if (nicNode == NULL) {
      WARN("The PCI node with busid %s was found, but no NIC sub-node was found.",busId.c_str());
      NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
      NCCLCHECK(xmlRemoveNode(head));
      memset(head, 0, sizeof(struct ncclXmlNode));
      return ncclSuccess;
    }

    // add net node if no exist
    NCCLCHECK(ncclTopoMappingGetNetNode(xml,nicNode,nicNameStr.c_str(),&netNode));
    // copy speed attr from head node if exists
    NCCLCHECK(xmlCopyAttr(netNode,head,"speed"));

    // Remove the head nodes that will not be used
    NCCLCHECK(xmlRemoveNode(head));
    memset(head, 0, sizeof(struct ncclXmlNode));
  }
  
  NCCLCHECK(xmlLoadSub(file, xml, netNode, NULL, 0));
  return ncclSuccess;
}

/*
 * 函数功能：按网卡名称查找net节点，若找到则返回该节点，否则新增一个net节点并设置name属性。用于nic标签处理时的net节点获取
 * 参数说明：
 *   xml: 输入参数，XML解析上下文结构体
 *   parent: 输入参数，父节点，若无法找到net节点则在父节点下新增一个net节点
 *   netNameAttr：输入参数，网卡名（如mlx5_0\shca_0）,即net节点的name属性值
 *   netNode：输出参数，net节点
 * 返回值：ncclSuccess表示成功，其他错误码表示解析/处理失败
 */
ncclResult_t ncclTopoMappingGetNetNode(struct ncclXml* xml, struct ncclXmlNode* parent, const char* netNameAttr ,struct ncclXmlNode** netNode) {
  if (!xml || !parent || !netNameAttr || !netNode) {
    return ncclSuccess;
  }
  
  // add net node if not exist
  NCCLCHECK(xmlGetSubKv(parent, "net", netNode, "name", netNameAttr));
  if (*netNode == NULL) {
    NCCLCHECK(XmlAddAndCopyNode(xml, parent, "net", netNode));
    (*netNode)->type = NODE_TYPE_SINGLE;

    NCCLCHECK(xmlSetAttr(*netNode, "name", netNameAttr));
  }
  return ncclSuccess;
}

/*
 * 函数功能：按属性名从源节点中查找属性值，并复制到目标节点。用于gpu和nic标签处理时的节点间属性拷贝
 * 参数说明：
 *   destNode: 输出参数，待复制属性的目标节点
 *   srcNode: 输入参数，源节点
 *   attrName：输入参数，属性名
 * 返回值：ncclSuccess表示成功，其他错误码表示解析/处理失败
 */
ncclResult_t xmlCopyAttr(struct ncclXmlNode* destNode, struct ncclXmlNode* srcNode,const char* attrName) {
  if (!destNode || !srcNode || !attrName ) return ncclSuccess;

  const char* value = NULL;
  NCCLCHECK(xmlGetAttr(srcNode, attrName, &value));
  if (value) {
    NCCLCHECK(xmlSetAttr(destNode, attrName, value));
  }
  return ncclSuccess;
}

/*
 * 函数功能：在parent节点下添加子节点，并按照标识拷贝源节点属性。用于gpu、nic、slot标签处理时的结构补充
 * 参数说明：
 *   xml: 输入参数，XML解析上下文结构体
 *   parent: 输入参数，父节点
 *   subNodeName：输入参数，子节点名
 *   subNode：输出参数，新增后的子节点
 *   srcNode：输入参数，待拷贝属性的源节点
 *   isCopy：输入参数，是否拷贝属性标识
 * 返回值：ncclSuccess表示成功，其他错误码表示解析/处理失败
 */
ncclResult_t XmlAddAndCopyNode(struct ncclXml* xml, struct ncclXmlNode* parent, const char* subNodeName, struct ncclXmlNode** subNode, struct ncclXmlNode* srcNode, bool isCopy) {
  if (!xml || !parent || !subNodeName || !subNode) {
    WARN("XML Topology mapping has wrong, creation and replication of nodes failed");
    return ncclInvalidUsage;
  }
  NCCLCHECK(xmlAddNode(xml, parent, subNodeName, subNode));

  if (srcNode && isCopy) {
    (*subNode)->nAttrs = srcNode->nAttrs;
    for (int i=0; i<srcNode->nAttrs; i++) {
      memcpy((*subNode)->attrs[i].key, srcNode->attrs[i].key, sizeof(srcNode->attrs[i].key));
      memcpy((*subNode)->attrs[i].value, srcNode->attrs[i].value, sizeof(srcNode->attrs[i].value));
    }
  }
  return ncclSuccess;
}

/*
 * 函数功能：按传入的slot号，以及全局的busid集合（g_busIdSet），获取对应的设备类型和编号
 * 参数说明：
 *   slot: 输入参数，slot号
 *   gpuId: 输出参数，设备GPU编号，当为SLOT_NODE_TYPE_NONE时表示非GPU设备或无法确定设备编号
 *   nicNameList: 输出参数，设备NIC名称列表，当为空时表示非NIC设备或无法确定设备名称
 *   classId：输出参数，设备（GPU、NIC）类型，当为SLOT_DEVICE_ID_NONE时表示无法确定设备类型
 * 返回值：ncclSuccess表示成功，其他错误码表示解析/处理失败
 */
ncclResult_t ncclTopoGetSlotInfo(int slot, int& gpuId, std::vector<std::string>& nicNameList, int& classId) {
  classId = SLOT_NODE_TYPE_NONE;
  gpuId = SLOT_DEVICE_ID_NONE;
  // Get the bus ID by slot number.
  char addressFilePath[PATH_MAX] = {};
  snprintf(addressFilePath,sizeof(addressFilePath),"/sys/bus/pci/slots/%d/address",slot);
  FILE* file = fopen(addressFilePath, "r");
  if (file == NULL) {
    WARN("Could not open slot address file %s : %s", addressFilePath, strerror(errno));
    return ncclSuccess;
  }
  // busid: xxxx:xx:xx.x
  // busid prefix : xxxx:xx:xx
  const size_t BUS_ID_PREFIX_SIZE = 10;
  const size_t BUS_ID_BUFFER_SIZE = BUS_ID_PREFIX_SIZE + 1;
  char busIdPrefix[BUS_ID_BUFFER_SIZE] = {};
  size_t bytes_read = fread(busIdPrefix, 1, BUS_ID_PREFIX_SIZE, file);
  fclose(file);
  if (bytes_read != BUS_ID_PREFIX_SIZE) {
    WARN("Could not read busid from address file %s", addressFilePath);
    return ncclSuccess;
  }
  busIdPrefix[BUS_ID_PREFIX_SIZE] = '\0';

  // Query the node by busid
  // test nic
  std::map<std::string, std::string>::iterator it = g_busIdSet.nicBusId.begin();
  while(it != g_busIdSet.nicBusId.end()) {
    if (strncmp(busIdPrefix, it->second.c_str(), BUS_ID_PREFIX_SIZE) == 0) {
      classId = SLOT_NODE_TYPE_NIC;
      // save all nic name that match the busid,Because it is possible that multiple nic share the same slot number.
      nicNameList.push_back(it->first);
    }
    it++;
  }

  // test gpu if no nic found
  // Start iterating from 1, skipping the innermost "busid"
  for(int i = 1; i<g_busIdSet.gpuBusId.size() && SLOT_NODE_TYPE_NONE == classId; i++) {
    std::map<std::string, std::string>::iterator it = g_busIdSet.gpuBusId[i].begin();
    while(it != g_busIdSet.gpuBusId[i].end()) {
      if (strncmp(busIdPrefix, it->second.c_str(), BUS_ID_PREFIX_SIZE) == 0) {
        classId = SLOT_NODE_TYPE_GPU;
        sscanf(it->first.c_str(), MAPPING_GPU_KEY_PREFIX"%d", &gpuId);
        break;
      }
      it++;
    }
  }

  switch (classId)
  {
  case SLOT_NODE_TYPE_GPU:
    INFO(NCCL_GRAPH, "slot id:%d - gpu dev:%d\n",slot,gpuId);
    break;
  case SLOT_NODE_TYPE_NIC:
  {
    std::string nicNameStr;
    for (int i = 0; i < nicNameList.size(); i++){
      nicNameStr+=" | " +nicNameList[i];
    }
    INFO(NCCL_GRAPH, "slot id:%d - nic id:%s\n",slot,nicNameStr.c_str());
    break;
  }
  default:
    WARN("slot id:%d - not found\n",slot);
    break;
  }
  return ncclSuccess;
}

/*
 * 函数功能：解析slot标签，读取slot的id属性，并按slot id确定的设备类型和编号调用对应的设备标签处理函数（ncclTopoXmlGraphLoadMappingGpu，ncclTopoXmlGraphLoadMappingNic）
 * 参数说明：
 *   file: 输入参数，配置文件
 *   xml: 输出参数，XML解析上下文结构体
 *   head：父节点
 * 返回值：ncclSuccess表示成功，其他错误码表示解析/处理失败
 */
ncclResult_t ncclTopoXmlGraphLoadMappingSlot(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  int slotId = 0;
  NCCLCHECK(xmlGetAttrIntDefault(head, "id", &slotId, -1));
  if (-1 == slotId) {
    WARN( "Cannot get slot id from xml file.");
    NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
    return ncclSuccess;
  }

  // Get the dev info by slot number.
  int gpuId = SLOT_DEVICE_ID_NONE,classId=SLOT_NODE_TYPE_NONE;
  std::vector<std::string> nicNameList;
  NCCLCHECK(ncclTopoGetSlotInfo(slotId, gpuId, nicNameList, classId));

  // Distribute by type
  if (SLOT_NODE_TYPE_GPU == classId) {
    // gpu type
    NCCLCHECK(xmlSetAttrInt(head,"dev",gpuId));
    NCCLCHECK(xmlUnsetAttr(head,"id"));
    NCCLCHECK(ncclTopoXmlGraphLoadMappingGpu(file,xml,head));
  } else if (SLOT_NODE_TYPE_NIC == classId) { 
    // nic type
    // multiple network cards may share the same slot number,need to be set up one by one
    for ( int i = 0; i < nicNameList.size(); i++) {
      //if not last one,then create new nic node and copy all attrs to simulate multiple NIC labels.
      struct ncclXmlNode* nicNode;
      if (i != nicNameList.size()-1){
        NCCLCHECK(XmlAddAndCopyNode(xml, head->parent, "nic", &nicNode, head, true));
        nicNode->type = NODE_TYPE_SINGLE;
      } else {
        nicNode = head;
      }
      NCCLCHECK(xmlSetAttr(nicNode,"id",nicNameList[i].c_str()));
      NCCLCHECK(ncclTopoXmlGraphLoadMappingNic(file,xml,nicNode));
    }
  } else {
    // none type
    WARN("Cannot turn the slot number[%d] into a GPU or NIC dev.",slotId);
    NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
    // The incorrect "slot" tags will be uniformly deleted later.
  }

  return ncclSuccess;
}

/*
 * 函数功能：收集全部pcieSwitch、nic、gpu节点的busid
 * 参数说明：
 *   comm: 输入参数，通讯组指针
 *   busIds: 输出参数，busid集合对象
 * 返回值：ncclSuccess表示成功，其他错误码表示解析/处理失败
 */
ncclResult_t collectNodesBusIds(struct ncclComm* comm) {
  //gpu
  g_busIdSet.gpuBusId.emplace_back();
  uint32_t deviceCnt = 0;
  NCCLCHECK(rocm_smi_getNumDevice(&deviceCnt));
  for (uint32_t i = 0; i < deviceCnt; i++) {
    char innerBusId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE] = {};
    NCCLCHECK(rocm_smi_getDevicePciBusIdString(i, innerBusId, sizeof(innerBusId)));
    std::string gpuKey = MAPPING_GPU_KEY_PREFIX+std::to_string(i);
    g_busIdSet.gpuBusId[0][gpuKey]=innerBusId;
    char* path = NULL;
    getPciPath(innerBusId, &path);
    int layerCount = 1;
    if (path) {
      int slashCount = 0;
      int parentOffset;
      for (parentOffset = strlen(path)-1; parentOffset > 0; parentOffset--) {
        if (path[parentOffset] == '/') {
          slashCount++;
          path[parentOffset] = '\0';
          int start = parentOffset - 1;
          while (start > 0 && path[start] != '/') start--;
          // Check whether the parent path looks like "BBBB:BB:DD.F" or not.
          if (checkBDFFormat(path+start+1) == 0) {
            // save last busid for pcieSwitch
            g_busIdSet.pcieSwitchBusId.emplace_back(g_busIdSet.gpuBusId[layerCount-1][gpuKey]);
            break;
          } else {
            if (slashCount % 2 == 0) {
              while(g_busIdSet.gpuBusId.size() <= layerCount) {
                g_busIdSet.gpuBusId.emplace_back();
              }
              char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE] = {};
              strncpy(busId, path+start+1, sizeof(busId)-1);
              g_busIdSet.gpuBusId[layerCount++][gpuKey]=busId;
            }
          }
        }
      }
      free(path);
    }

    std::string busIdListStr;
    for (int vecterIndex = 0; vecterIndex < layerCount; vecterIndex++) {
      if (g_busIdSet.gpuBusId[vecterIndex].find(gpuKey) != g_busIdSet.gpuBusId[vecterIndex].end()) {
        busIdListStr += " | " + g_busIdSet.gpuBusId[vecterIndex][gpuKey];
      }
    }
    
    INFO(NCCL_GRAPH, "gpu key:%s busIdList:%s \n",gpuKey.c_str(),busIdListStr.c_str());
  }

  std::string pcieSwitchListStr;
  for(int i=0;i<g_busIdSet.pcieSwitchBusId.size();i++) {
    pcieSwitchListStr += " | " + g_busIdSet.pcieSwitchBusId[i];
  }
  INFO(NCCL_GRAPH, "pcieSwitch busIdList :%s \n",pcieSwitchListStr.c_str());

  //nic
  int netDevCount;
  // note: netDevCount only include valid network devices. If the status of the network device is "Down", it will not be counted.
  NCCLCHECK(comm->ncclNet->devices(&netDevCount));
  for (int i = 0; i < netDevCount; i++) { 
    ncclNetProperties_t props;
    NCCLCHECK(comm->ncclNet->getProperties(i, &props));
    if (props.pciPath == NULL || props.name == NULL) {
      WARN("Unable to obtain the pciPath[%s] or name[%s] attributes from NetProperties.",props.pciPath?props.pciPath : "NULL", props.name?props.name : "NULL");
      continue;
    }
    char* pciSysPath = props.pciPath;
    int offset;
    for (offset=strlen(pciSysPath)-1; pciSysPath[offset] != '/'; offset--);
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE] = {};
    if (checkBDFFormat(pciSysPath+offset+1) == 1) {
      strncpy(busId, pciSysPath+offset+1, sizeof(busId)-1);
      g_busIdSet.nicBusId[props.name]=busId;
      INFO(NCCL_GRAPH, "nic name:%s busId:%s \n",props.name,busId);
    } else {
      WARN("The pciPath[%s] attribute of the network device[%s] is not in the correct format.", props.pciPath, props.name);
    }
  }

  return ncclSuccess;
} 

// Return Prefix string
inline std::string envTypeInfo::getPrefixFromNicName(const std::string& nicName) {
    // 1. First, locate the underline.
    size_t pos = nicName.find('_');
    if (pos == std::string::npos)
        // 2. No underline. Find the first number.
        pos = nicName.find_first_of("0123456789");

    return pos == std::string::npos ? nicName   // Nothing at all. Back to the whole.
                                    : nicName.substr(0, pos);
}

inline void envTypeInfo::readLineFromFile(const std::string& filePath , std::string& lineContent) {
    std::ifstream file(filePath);
    lineContent.clear();
    if (file.is_open()) {
      std::getline(file, lineContent);
      file.close();
    }
}

inline int envTypeInfo::readInfinibandRate(const std::string& filePath) {
    std::string line;  
    readLineFromFile(filePath, line);
    if (line.empty()) {
        return 0;
    }
    
    int rate;
    if (strstr(line.c_str(), "Gb/sec")) {
      const char* format = (line[0] >= '0' && line[0] <= '9') ? "%d" : "%*[^0-9]%d";
      if (sscanf(line.c_str(), format, &rate) == 1) {
        return rate;
      }
    }
    return 0;
}

// Get NIC type, quantity and rate from /sys/class/infiniband
int envTypeInfo::getNicInfoFromSys(std::string& nicNamePrefix, std::string& nicType, std::string& nicRateList) { 
  std::string path = "/sys/class/infiniband";
  DIR* dir = opendir(path.c_str());
  if (!dir) {
    return 0;
  }

  std::vector<std::string> devices;
  dirent* entry;
  while ((entry = readdir(dir))) {
    if (entry->d_name[0] != '.')          // 跳过 "."  ".."
      devices.emplace_back(entry->d_name);
  }
  closedir(dir);

  if (!devices.empty()) {
    // sort device names
    std::sort(devices.begin(), devices.end());

    // Get NIC Name prefix
    nicNamePrefix = getPrefixFromNicName(devices[0]);

    // Get NIC type
    readLineFromFile(path + "/" + devices[0] + "/ports/1/link_layer", nicType);

    // Get NIC rate list
    for(int i = 0; i < devices.size(); i++){
      int rate = readInfinibandRate(path + "/" + devices[i] + "/ports/1/rate");
      if (i == 0) {
        nicRateList += std::to_string(rate);
      } else {
        nicRateList += "-" + std::to_string(rate);
      }
    }
  }

  return devices.size();
}

// Get Hylink type, max group id, group size from rocm-smi
void envTypeInfo::getHylinkInfo(int gpuCount, int& hyLinkType, int& hyLinkMaxGroupId, int& hyLinkGroupSize, std::string& hyLinkInfoStr){
  #ifdef RSMI_SUPPORT_XHCL_TOPO
  if (gpuCount <= 0) {
    return;
  }
  uint32_t linkNum;
  uint32_t linkState;
  for (int rocmDev = 0; rocmDev < gpuCount; rocmDev++) {
    // linkNum may be 0 when no hylink
    NCCLCHECK(rocm_smi_dev_xhcl_link_number_get(rocmDev, &linkNum));
    if (linkNum <= 0) {
      // no hylink
      break;
    }

    // Used to identify the type of the opposite end node : 0-GPU 1-SWITCH 
    // run only once when rocmDev == 0
    if (rocmDev == 0) {
      RSMI_XHCL_LINK_TYPE devType = RSMI_XHCL_LINK_TYPE_GPU;
      NCCLCHECK(rocm_smi_dev_xhcl_link_remote_dev_type_get(rocmDev, 0, &devType));
      hyLinkType = devType;
    }

    // get max group id
    // linkState may be 0 when no hylink, otherwise is group id of link
    NCCLCHECK(rocm_smi_dev_xhcl_link_state_get(rocmDev, 0, &linkState));
    if (linkState <= 0) {
      // no hylink
      break;
    } else if (hyLinkMaxGroupId < linkState) {
      hyLinkMaxGroupId = linkState;
    } 
  }

  // only when linkNum and hyLinkMaxGroupId are both greater than 0, return valid hyLinkInfoStr
  if (linkNum > 0 && hyLinkMaxGroupId > 0) {
    hyLinkGroupSize = linkNum / hyLinkMaxGroupId;
    hyLinkInfoStr = std::to_string(hyLinkType) + "_" + std::to_string(hyLinkGroupSize) + "_" + std::to_string(hyLinkMaxGroupId);
  }
  #endif
  return;
}

// Collect environmental information, including GPU architecture, GPU quantity, CPU architecture, NIC type and NIC quantity
void envTypeInfo::_init() {
  // gpu gcn
  char gcnArchNameSubstr[128]={};
  GetGcnArchName(0,gcnArchNameSubstr);
  gpuGcn = gcnArchNameSubstr;
  // gpu count
  NCCLCHECK(rocm_smi_getNumDevice(&gpuCount));

  //cpu arch 
  #if defined(__PPC__)
    cpuArch = "ppc64";
  #elif defined(__aarch64__)
    cpuArch = "arm64";
  #elif defined(__x86_64__)
    cpuArch = "x86_64";
  #endif

  //cpu vendor
  #if defined(__x86_64__)
    union {
      struct {
        // CPUID 0 String register order
        uint32_t ebx;
        uint32_t edx;
        uint32_t ecx;
      };
      char vendor[12];
    } cpuid0;

    unsigned unused;
    __cpuid(0, unused, cpuid0.ebx, cpuid0.ecx, cpuid0.edx);
    cpuVendor = std::string(cpuid0.vendor, 12); 
  #endif

  //NIC type, quantity and rate from /sys/class/infiniband
  nicCount = getNicInfoFromSys(nicNamePrefix, nicType, nicRateList);

  if (!cpuVendor.empty()) {
    envTypeStr = gpuGcn + "_" + std::to_string(gpuCount) + "_" +
                cpuArch + "_" + cpuVendor + "_" + 
                nicNamePrefix + "_" + std::to_string(nicCount) + "_" + nicType + "_" + nicRateList;
  } else {
    envTypeStr = gpuGcn + "_" + std::to_string(gpuCount) + "_" +
                cpuArch + "_" + nicNamePrefix + "_" + std::to_string(nicCount) + "_" + nicType + "_" + nicRateList;
  }
  
  //hylink type, max group id, group size
  std::string linkInfo; 
  getHylinkInfo(gpuCount, hyLinkType, hyLinkMaxGroupId, hyLinkGroupSize, linkInfo);

  // add hylink info if exists
  if (!linkInfo.empty()) {
    envTypeStr += "_" + linkInfo;
  }

  return;
}


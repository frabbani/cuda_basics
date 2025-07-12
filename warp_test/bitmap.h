#pragma once

#include <vector>

bool importBMP(std::vector<uint8_t>& rgbData, int& w, int& h, const char* filename);
bool exportBMP(const std::vector<uint8_t>&rgbData, int w, int h, const char * filename);

void encryptBMP(const std::vector<uint8_t>& rgbData, const char *key, std::vector<uint8_t>& rgbEncrypted);
void genPermTables(const char* key, size_t numValues, std::vector<uint32_t>& perm, std::vector<uint32_t>& inversePerm);
void decryptBMP(const std::vector<uint8_t>& rgbEncrypted, std::vector<uint8_t>& rgbData, const std::vector<uint32_t>& perm);
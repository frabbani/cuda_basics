#include "bitmap.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

enum Compression {
    RGB = 0,
    RLE8,
    RLE4,
    BITFIELDS,
    JPEG,
    PNG,
};

#pragma pack(push, 1)

//2 bytes
struct FileMagic {
    uint8_t num0, num1;
};

//12 bytes
struct FileHeader {
    uint32_t fileSize;
    uint16_t creators[2];
    uint32_t dataOffset;
};

//40 bytes, all windows versions since 3.0
struct DibHeader {
    uint32_t headerSize;
    int32_t width, height;
    uint16_t numPlanes, bitsPerPixel;
    uint32_t compression;
    uint32_t dataSize;
    int32_t hPixelsPer, vPixelsPer;  //horizontal and vertical pixels-per-meter
    uint32_t numPalColors, numImportantColors;
};


bool exportBMP(const std::vector<uint8_t>& rgbData, int w, int h, const char* filename) {
    if (rgbData.size() != (w * h * 3)) {
        printf("%s - invalid data size\n", __FUNCTION__);
        return false;
    }

    // Add ".bmp" if not already there
    char file[256];
    strncpy(file, filename, sizeof(file));
    file[sizeof(file) - 1] = '\0';
    if (strstr(file, ".bmp") == nullptr)
        strcat(file, ".bmp");

    FILE* fp = fopen(file, "wb");
    if (!fp) {
        perror("fopen");
        return false;
    }

    // Padding per row (rows must be aligned to 4 bytes)
    int rowSize = w * 3;
    int padding = (4 - (rowSize % 4)) % 4;
    int paddedRowSize = rowSize + padding;
    int dataSize = paddedRowSize * h;

    // --- File Header ---
    FileMagic magic = { 'B', 'M' };
    fwrite(&magic, 2, 1, fp);

    FileHeader fileHeader;
    fileHeader.fileSize = 14 + 40 + dataSize; // file header + DIB header + pixel data
    fileHeader.creators[0] = fileHeader.creators[1] = 0;
    fileHeader.dataOffset = 14 + 40;
    fwrite(&fileHeader, sizeof(fileHeader), 1, fp);

    // --- DIB Header ---
    DibHeader dibHeader;
    dibHeader.headerSize = 40;
    dibHeader.width = w;
    dibHeader.height = h;
    dibHeader.numPlanes = 1;
    dibHeader.bitsPerPixel = 24;
    dibHeader.compression = RGB;
    dibHeader.dataSize = dataSize;
    dibHeader.hPixelsPer = dibHeader.vPixelsPer = 1000;
    dibHeader.numPalColors = dibHeader.numImportantColors = 0;
    fwrite(&dibHeader, sizeof(DibHeader), 1, fp);

    // --- Pixel Data (bottom-up) ---
    for (int y = h - 1; y >= 0; y--) {
        const uint8_t* row = &rgbData[y * w * 3];
        fwrite(row, 1, rowSize, fp);
        for (int i = 0; i < padding; i++)
            fputc(0x00, fp); // zero padding
    }

    fclose(fp);
    printf("%s - bitmap file '%s' created\n", __FUNCTION__, file);
    return true;
}
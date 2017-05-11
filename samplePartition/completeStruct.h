#ifndef COMPLETE_STRUCT_H
#define COMPLETE_STRUCT_H

typdef struct complete{
    void* cpuMemory;
    void* gpuMemory;
    int gpuAColumns;
    int ARows;
    int cpuAStart;
    int cpuAEnd;
} memoryLocations;

#endif
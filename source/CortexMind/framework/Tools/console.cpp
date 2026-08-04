//
// Created by muham on 4.08.2026.
//

#include "CortexMind/framework/Tools/console.hpp"
#if _WIN32
    #include <windows.h>
#endif //#if _WIN32

using namespace cortex::_fw;

void detail::EnableVirtualTerminal() {
    #if _WIN32
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);

        DWORD mode = 0;
        GetConsoleMode(hOut, &mode);

        mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;

        SetConsoleMode(hOut, mode);
    #endif //#if _WIN32
}

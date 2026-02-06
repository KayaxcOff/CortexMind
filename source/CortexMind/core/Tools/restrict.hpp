//
// Created by muham on 2.02.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_RESTRICT_HPP
#define CORTEXMIND_CORE_TOOLS_RESTRICT_HPP

#if defined(_MSC_VER)
  #define restrict __restrict
#elif defined(__GNUC__) || defined(__clang__)
  #define restrict __restrict__
#else
  #define restrict
#endif


#endif //CORTEXMIND_CORE_TOOLS_RESTRICT_HPP
#pragma once
#ifndef _COLORS_H_
#define _COLORS_H_


/*
┌───────────────────────────┐
│     COLOR TABLE           │
│ Name           │ FG │ BG  │
│───────────────────────────│
│ Black          │ 30 │ 40  │
│ Red            │ 31 │ 41  │
│ Green          │ 32 │ 42  │
│ Yellow         │ 33 │ 43  │
│ Blue           │ 34 │ 44  │
│ Magenta        │ 35 │ 45  │
│ Cyan           │ 36 │ 46  │
│ White          │ 37 │ 47  │
│ Bright Black   │ 90 │ 100 │
│ Bright Red     │ 91 │ 101 │
│ Bright Green   │ 92 │ 102 │
│ Bright Yellow  │ 93 │ 103 │
│ Bright Blue    │ 94 │ 104 │
│ Bright Magenta │ 95 │ 105 │
│ Bright Cyan    │ 96 │ 106 │
│ Bright White   │ 97 │ 107 │
└───────────────────────────┘
*/

//----------------MODIFIERS-----------------------
#ifndef NON_GENERAL_COLORS_NAMES
#  define UNDERLINE			"\033[4m"
#  define DOUBLE_UNDERLINE		"\033[21m"
#  define LINE_THROUGH			"\033[9m"
#  define ITALIC			"\033[3m"
#else
#  define UNDERLINE_TEXT_MOD		"\033[4m"
#  define DOUBLE_UNDERLINE_TEXT_MOD	"\033[21m"
#  define LINE_THROUGH_TEXT_MOD		"\033[9m"
#  define ITALIC_TEXT_MOD		"\033[3m"
#endif

//----------------SPECIFIERS----------------------
#ifndef NON_GENERAL_COLORS_NAMES
#  define RESET_COLOR			"\033[0m"
#  define TRANSPARENT_COLOR		"\033[2m"
#else
#  define RESET_COLOR_TEXT_SPEC		"\033[0m"
#  define TRANSPARENT_COLOR_TEXT_SPEC 	"\033[2m"
#endif

//----------------COLORS--------------------------
#ifndef NON_GENERAL_COLORS_NAMES
#  define BLACK				"\033[30m"
#  define BRIGHT_BLACK			"\033[90m"
#  define BLACK_BG			"\033[40m"
#  define BRIGHT_BLACK_BG		"\033[100m"
#else
#  define BLACK_TEXT_CL			"\033[30m"
#  define BRIGHT_BLACK_TEXT_CL		"\033[90m"
#  define BLACK_BG_TEXT_CL		"\033[40m"
#  define BRIGHT_BLACK_BG_TEXT_CL 	"\033[100m"
#endif

#ifndef NON_GENERAL_COLORS_NAMES
#  define RED				"\033[31m"
#  define BRIGHT_RED			"\033[91m"
#  define RED_BG			"\033[41m"
#  define BRIGHT_RED_BG			"\033[101m"
#else
#  define RED_TEXT_CL			"\033[31m"
#  define BRIGHT_RED_TEXT_CL		"\033[91m"
#  define RED_BG_TEXT_CL		"\033[41m"
#  define BRIGHT_RED_BG_TEXT_CL		"\033[101m"
#endif

#ifndef NON_GENERAL_COLORS_NAMES
#  define GREEN				"\033[32m"
#  define BRIGHT_GREEN			"\033[92m"
#  define GREEN_BG			"\033[42m"
#  define BRIGHT_GREEN_BG		"\033[102m"
#else
#  define GREEN_TEXT_CL			"\033[32m"
#  define BRIGHT_GREEN_TEXT_CL		"\033[92m"
#  define GREEN_BG_TEXT_CL		"\033[42m"
#  define BRIGHT_GREEN_BG_TEXT_CL 	"\033[102m"
#endif

#ifndef NON_GENERAL_COLORS_NAMES
#  define YELLOW			"\033[33m"
#  define BRIGHT_YELLOW			"\033[93m"
#  define YELLOW_BG			"\033[43m"
#  define BRIGHT_YELLOW_BG		"\033[103m"
#else
#  define YELLOW_TEXT_CL		"\033[33m"
#  define BRIGHT_YELLOW_TEXT_CL		"\033[93m"
#  define YELLOW_BG_TEXT_CL		"\033[43m"
#  define BRIGHT_YELLOW_BG_TEXT_CL	"\033[103m"
#endif

#ifndef NON_GENERAL_COLORS_NAMES
#  define BLUE				"\033[34m"
#  define BRIGHT_BLUE			"\033[94m"
#  define BLUE_BG			"\033[44m"
#  define BRIGHT_BLUE_BG		"\033[104m"
#else
#  define BLUE_TEXT_CL			"\033[34m"
#  define BRIGHT_BLUE_TEXT_CL		"\033[94m"
#  define BLUE_BG_TEXT_CL		"\033[44m"
#  define BRIGHT_BLUE_BG_TEXT_CL	"\033[104m"
#endif

#ifndef NON_GENERAL_COLORS_NAMES
#  define BLUE				"\033[34m"
#  define MAGENTA			"\033[35m"
#  define BRIGHT_MAGENTA		"\033[95m"
#  define MAGENTA_BG			"\033[45m"
#  define BRIGHT_MAGENTA_BG		"\033[105m"
#else
#  define BLUE_TEXT_CL			"\033[34m"
#  define MAGENTA_TEXT_CL		"\033[35m"
#  define BRIGHT_MAGENTA_TEXT_CL	"\033[95m"
#  define MAGENTA_BG_TEXT_CL		"\033[45m"
#  define BRIGHT_MAGENTA_BG_TEXT_CL	"\033[105m"
#endif

#ifndef NON_GENERAL_COLORS_NAMES
#  define CYAN				"\033[36m"
#  define BRIGHT_CYAN			"\033[96m"
#  define CYAN_BG			"\033[46m"
#  define BRIGHT_CYAN_BG		"\033[106m"
#else
#  define CYAN_TEXT_CL			"\033[36m"
#  define BRIGHT_CYAN_TEXT_CL		"\033[96m"
#  define CYAN_BG_TEXT_CL		"\033[46m"
#  define BRIGHT_CYAN_BG_TEXT_CL	"\033[106m"
#endif

#ifndef NON_GENERAL_COLORS_NAMES
#  define WHITE				"\033[37m"
#  define BRIGHT_WHITE			"\033[97m"
#  define WHITE_BG			"\033[47m"
#  define BRIGHT_WHITE_BG		"\033[107m"
#else
#  define WHITE_TEXT_CL			"\033[37m"
#  define BRIGHT_WHITE_TEXT_CL		"\033[97m"
#  define WHITE_BG_TEXT_CL		"\033[47m"
#  define BRIGHT_WHITE_BG_TEXT_CL	"\033[107m"
#endif


#endif

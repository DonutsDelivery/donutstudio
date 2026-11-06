# Writes a compiled GPU kernel (CUDA PTX or a HIP code object) into a C++ file as
# a byte array with a trailing zero, so PTX also reads as a C string.
# Run with -DIN=<input> -DOUT=<output> -DNAME=<symbol>.
file(READ "${IN}" hex HEX)
string(LENGTH "${hex}" digits)
math(EXPR size "${digits} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
string(REGEX REPLACE "((0x..,){32})" "\\1\n" bytes "${bytes}")
file(WRITE "${OUT}" "#include <cstddef>\nextern const unsigned char ${NAME}[];\nextern const std::size_t ${NAME}Size;\nconst unsigned char ${NAME}[] = {\n${bytes}0x00 };\nconst std::size_t ${NAME}Size = ${size};\n")

# Writes the kernel's PTX, which is text, into a C++ file as raw string chunks.
# Run with -DPTX=<input> -DOUT=<output>. Chunks keep each literal under the
# length every compiler accepts.
file(READ "${PTX}" text)
string(LENGTH "${text}" size)
set(chunk 16000)
set(out "#include <cstddef>\nextern const char* const kArbitOptixKernelPtx[];\nextern const std::size_t kArbitOptixKernelPtxChunks;\nconst char* const kArbitOptixKernelPtx[] = {\n")
set(count 0)
set(at 0)
while(at LESS size)
    string(SUBSTRING "${text}" ${at} ${chunk} piece)
    string(APPEND out "R\"PTXCHUNK(${piece})PTXCHUNK\",\n")
    math(EXPR at "${at} + ${chunk}")
    math(EXPR count "${count} + 1")
endwhile()
string(APPEND out "};\nconst std::size_t kArbitOptixKernelPtxChunks = ${count};\n")
file(WRITE "${OUT}" "${out}")

#include <kvazaar.h>
#if KVZ_BIT_DEPTH != 10
#error The private Kvazaar library must compile at exactly ten bits.
#endif

int kvz_adapter_compiled_bit_depth(void)
{
    return KVZ_BIT_DEPTH;
}

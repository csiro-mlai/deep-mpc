/*
 * BufferScope.h
 *
 */

#ifndef PROTOCOLS_BUFFERSCOPE_H_
#define PROTOCOLS_BUFFERSCOPE_H_

template<class T> class BufferPrep;
template<class T> class Preprocessing;

template<class T>
class BufferScope
{
    BufferPrep<T>& prep;
    int bak;

public:
    BufferScope(Preprocessing<T> & prep, int buffer_size) :
            prep(dynamic_cast<BufferPrep<T>&>(prep))
    {
        bak = this->prep.buffer_size;
        this->prep.buffer_size = buffer_size;
    }

    ~BufferScope()
    {
        prep.buffer_size = bak;
    }
};

#endif /* PROTOCOLS_BUFFERSCOPE_H_ */

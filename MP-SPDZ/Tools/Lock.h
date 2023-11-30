/*
 * Lock.h
 *
 */

#ifndef TOOLS_LOCK_H_
#define TOOLS_LOCK_H_

#include <pthread.h>

class Lock
{
    pthread_mutex_t mutex;
public:
    Lock();
    virtual ~Lock();

    void lock();
    void unlock();
};

class ScopeLock
{
    Lock& lock;

public:
    ScopeLock(Lock& lock) :
            lock(lock)
    {
        lock.lock();
    }

    ~ScopeLock()
    {
        lock.unlock();
    }
};

#endif /* TOOLS_LOCK_H_ */

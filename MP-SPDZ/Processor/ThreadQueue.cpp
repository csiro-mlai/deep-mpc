/*
 * ThreadQueue.cpp
 *
 */


#include "ThreadQueue.h"

thread_local ThreadQueue* ThreadQueue::thread_queue = 0;

void ThreadQueue::schedule(const ThreadJob& job)
{
    lock.lock();
    left++;
#ifdef DEBUG_THREAD_QUEUE
        cerr << this << ": " << left << " left" << endl;
#endif
    lock.unlock();
    if (thread_queue)
        thread_queue->wait_timer.start();
    in.push(job);
    if (thread_queue)
        thread_queue->wait_timer.stop();
}

ThreadJob ThreadQueue::next()
{
    return in.pop();
}

void ThreadQueue::finished(const ThreadJob& job)
{
    out.push(job);
}

void ThreadQueue::finished(const ThreadJob& job, const NamedCommStats& new_comm_stats)
{
    finished(job);
    set_comm_stats(new_comm_stats);
}

void ThreadQueue::set_comm_stats(const NamedCommStats& new_comm_stats)
{
    lock.lock();
    comm_stats = new_comm_stats;
    lock.unlock();
}

ThreadJob ThreadQueue::result()
{
    if (thread_queue)
        thread_queue->wait_timer.start();
    auto res = out.pop();
    if (thread_queue)
        thread_queue->wait_timer.stop();
    lock.lock();
    left--;
#ifdef DEBUG_THREAD_QUEUE
        cerr << this << ": " << left << " left" << endl;
#endif
    lock.unlock();
    return res;
}

NamedCommStats ThreadQueue::get_comm_stats()
{
    lock.lock();
    auto res = comm_stats;
    lock.unlock();
    return res;
}

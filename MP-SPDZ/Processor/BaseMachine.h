/*
 * BaseMachine.h
 *
 */

#ifndef PROCESSOR_BASEMACHINE_H_
#define PROCESSOR_BASEMACHINE_H_

#include "Tools/time-func.h"
#include "Tools/TimerWithComm.h"
#include "OT/OTTripleSetup.h"
#include "ThreadJob.h"
#include "ThreadQueues.h"
#include "Program.h"
#include "OnlineOptions.h"

#include <map>
#include <fstream>
using namespace std;

void print_usage(ostream& o, const char* name, size_t capacity);

class BaseMachine
{
protected:
    static BaseMachine* singleton;

    static thread_local OnDemandOTTripleSetup ot_setup;

    std::map<int,TimerWithComm> timer;

    string compiler;
    string domain;
    string relevant_opts;

    virtual size_t load_program(const string& threadname,
            const string& filename);

public:
    static thread_local int thread_num;

    string progname;
    int nthreads;

    ThreadQueues queues;

    vector<string> bc_filenames;

    vector<Program> progs;

    static BaseMachine& s();
    static bool has_singleton() { return singleton != 0; }
    static bool has_program();

    static string memory_filename(const string& type_short, int my_number);

    static string get_domain(string progname);
    static int ring_size_from_schedule(string progname);
    static int prime_length_from_schedule(string progname);
    static bigint prime_from_schedule(string progname);

    template<class T>
    static int batch_size(Dtype type, int buffer_size = 0, int fallback = 0);
    template<class T>
    static int edabit_batch_size(int n_bits, int buffer_size = 0);
    static int edabit_bucket_size(int n_bits);

    BaseMachine();
    virtual ~BaseMachine() {}

    void load_schedule(const string& progname, bool load_bytecode = true);
    void print_compiler();

    void time();
    void start(int n);
    void stop(int n);

    void print_timers();

    virtual void reqbl(int) {}
    virtual void active(int) {}

    static OTTripleSetup fresh_ot_setup(Player& P);

    NamedCommStats total_comm();
    void set_thread_comm(const NamedCommStats& stats);

    void print_global_comm(Player& P, const NamedCommStats& stats);
    void print_comm(Player& P, const NamedCommStats& stats);

    virtual const Names& get_N() { throw not_implemented(); }
};

inline OTTripleSetup BaseMachine::fresh_ot_setup(Player& P)
{
    return ot_setup.get_fresh(P);
}

template<class T>
int BaseMachine::batch_size(Dtype type, int buffer_size, int fallback)
{
    int n_opts;
    int n = 0;
    int res = 0;

    if (buffer_size > 0)
        n_opts = buffer_size;
    else if (fallback > 0)
        n_opts = fallback;
    else
        n_opts = OnlineOptions::singleton.batch_size;

    if (buffer_size <= 0 and has_program())
    {
        auto files = s().progs[0].get_offline_data_used().files;
        auto usage = files[T::clear::field_type()];

        if (type == DATA_DABIT and T::LivePrep::bits_from_dabits())
            n = usage[DATA_BIT] + usage[DATA_DABIT];
        else if (type == DATA_BIT and T::LivePrep::dabits_from_bits())
            n = usage[DATA_BIT] + usage[DATA_DABIT];
        else
            n = usage[type];
    }
    else if (type != DATA_DABIT)
    {
        n = buffer_size;
        buffer_size = 0;
        n_opts = OnlineOptions::singleton.batch_size;
    }

    if (n > 0 and not (buffer_size > 0))
    {
        bool used_frac = false;
        if (n > n_opts)
        {
            // finding the right fraction
            for (int i = 1; i <= 10; i++)
            {
                int frac = DIV_CEIL(n, i);
                if (frac <= n_opts)
                {
                    res = frac;
                    used_frac = true;
#ifdef DEBUG_BATCH_SIZE
                    cerr << "found fraction " << frac << endl;
#endif
                    break;
                }
            }
        }
        if (not used_frac)
            res = min(n, n_opts);
    }
    else
        res = n_opts;

#ifdef DEBUG_BATCH_SIZE
    cerr << DataPositions::dtype_names[type] << " " << T::type_string()
            << " res=" << res << " n="
            << n << " n_opts=" << n_opts << " buffer_size=" << buffer_size << endl;
#endif

    assert(res > 0);
    return res;
}

template<class T>
int BaseMachine::edabit_batch_size(int n_bits, int buffer_size)
{
    int n_opts;
    int n = 0;
    int res;

    if (buffer_size > 0)
        n_opts = buffer_size;
    else
        n_opts = OnlineOptions::singleton.batch_size;

    if (has_program())
    {
        n = s().progs[0].get_offline_data_used().total_edabits(n_bits);
    }

    if (n > 0 and not (buffer_size > 0))
        res = min(n, n_opts);
    else
        res = n_opts;

#ifdef DEBUG_BATCH_SIZE
    cerr << "edaBits " << T::type_string() << " (" << n_bits
            << ") res=" << res << " n="
            << n << " n_opts=" << n_opts << " buffer_size=" << buffer_size << endl;
#endif

    assert(res > 0);
    return res;
}

#endif /* PROCESSOR_BASEMACHINE_H_ */

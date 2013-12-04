#pragma once

// Always define NOMINMAX before including windows.h to prevent
// strange errors wherever min() and max() are used in any context.
#define NOMINMAX
#include <windows.h>
#include <string>
#include <map>

using namespace std;

class PerfTimer
{
public:
	PerfTimer();
	PerfTimer(const string& timerName, int level = 0, const char *timerDescriptor = NULL);
	~PerfTimer();

	void start();
	void stop();
	void reset();
	double getTime() const;
	double getAverageTime() const;
	double getFPS() const;
	string getFormattedTime(bool isShowAverage = true) const;
	long getCount() const;
	int getLevel() const;

private:
	// Members
	float freq_;
	LARGE_INTEGER time1_;
	LARGE_INTEGER time2_;
	LONGLONG total_time_;
	long count_;
	bool is_running_;
	string name_;
	string desc_;
	int level_;
};

struct TimerNode
{
	PerfTimer timer;
	int count;
	int treeLevel;
	TimerNode(const PerfTimer& Timer, int Count, int TreeLevel )
	{
		timer = Timer;
		count = Count;
		treeLevel = TreeLevel;
	}
	TimerNode()
	{
		count = -1;
		treeLevel = -1;
	}
};

class CodeTimer
{
public:
	CodeTimer(void);
	~CodeTimer(void);
	void start(const string& timerName, int level = 0, char *timerDescriptor = NULL);
	void stop(const string& timerName);
	void printTimes(int minPrintLevel = 0, bool isShowAverage = false);
	void printTimeTree(int minPrintLevel = 0, bool isShowAverage = true);
	void clear();

private:
	typedef map<string, TimerNode > TimerMap;
	TimerMap timer_pool_;
	int counter_;
	int cur_tree_level_;
};

extern CodeTimer gCodeTimer;

#ifdef __CUDACC__
#define G_CODE_TIMER_BEFORE_STOP cudaDeviceSynchronize()
#else
#define G_CODE_TIMER_BEFORE_STOP 
#endif

#ifndef G_CODE_TIMER_BEFORE_START
#define G_CODE_TIMER_BEFORE_START
#endif
#ifndef G_CODE_TIMER_AFTER_START
#define G_CODE_TIMER_AFTER_START
#endif

#ifndef G_CODE_TIMER_BEFORE_STOP
#define G_CODE_TIMER_BEFORE_STOP
#endif
#ifndef G_CODE_TIMER_AFTER_STOP
#define G_CODE_TIMER_AFTER_STOP
#endif

#ifndef G_CODE_TIMER_TIMING_LEVEL
#define G_CODE_TIMER_TIMING_LEVEL 1
#endif

//#define GTB(x) do { \
//	G_CODE_TIMER_BEFORE_START; \
//	gCodeTimer.start((x)) \
//	G_CODE_TIMER_AFTER_START; \
//} while(0)

//#define GTE(x) do { \
//	G_CODE_TIMER_BEFORE_STOP; \
//	gCodeTimer.stop((x)); \
//	G_CODE_TIMER_AFTER_STOP; \
//} while(0)

#define CONCATE(name) __FUNCTION__##"_"##name

inline void GTB_RawName(const string& name, int level = 0)
{
//	if(level > G_CODE_TIMER_TIMING_LEVEL)
	{
		G_CODE_TIMER_BEFORE_START;
		gCodeTimer.start(name, level);
		G_CODE_TIMER_AFTER_START; 
	}
}

inline void GTE_RawName(const string &name)
{
//	if(level > G_CODE_TIMER_TIMING_LEVEL)
	{
		G_CODE_TIMER_BEFORE_STOP;
		gCodeTimer.stop(name);
		G_CODE_TIMER_AFTER_STOP;
	}
} 

//#define NO_CODE_TIMING

#ifdef NO_CODE_TIMING

#define GTB(x) 
#define GTE(x) 

#else
//#define GTB(x) GTB_RawName(CONCATE(x))
//
//#define GTE(x) GTE_RawName(CONCATE(x))

#define GTB(x) GTB_RawName(x)

#define GTE(x) GTE_RawName(x)
#endif

extern double total_time;
extern double total_count;
extern double total_fps;
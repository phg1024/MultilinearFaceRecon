#include "CodeTimer.h"
#include <cassert>

PerfTimer::PerfTimer( const string& timerName, int level, const char *timerDescriptor /*= NULL*/ )
{
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	freq_ = 1.0f / freq.QuadPart;
	total_time_ = 0;
	count_ = 0;
	is_running_ = false;
	name_ = timerName;
	level_ = level;
	if(timerDescriptor)
	{
		desc_ = string(timerDescriptor);
	}
}

PerfTimer::PerfTimer()
{

}

PerfTimer::~PerfTimer()
{

}

void PerfTimer::start()
{
	QueryPerformanceCounter(&time1_);
	is_running_ = true;
}

void PerfTimer::stop()
{
//	assert(is_running_);
	if(!is_running_)
	{
		printf("Timer %s is not running.\n", name_.c_str());
		return;
	}
	QueryPerformanceCounter(&time2_);
	total_time_ += (time2_.QuadPart - time1_.QuadPart);
	count_++;
	is_running_ = false;
}

double PerfTimer::getTime() const
{
	assert(!is_running_);
	return total_time_ * freq_;
}

std::string PerfTimer::getFormattedTime(bool isShowAverage) const
{
	string show_name = name_;
	if(!desc_.empty())
	{
		show_name = desc_;
	}
	char str[1000];
	if(isShowAverage) 
	{
		sprintf(str, "%s: %0.3lfs (%0.2lf fps, %0.3lfms)", show_name.c_str(), getTime(), getFPS(), getAverageTime() * 1000.0);
	} else
	{
		sprintf(str, "%s: %0.3lfs", show_name.c_str(), getTime());
	}
	string ret(str);
	return ret;
}

double PerfTimer::getAverageTime() const
{
	return getTime()/getCount();
}

long PerfTimer::getCount() const
{
	assert(!is_running_);
	return count_;
}

void PerfTimer::reset()
{
	total_time_ = 0;
	is_running_ = false;
	count_ = 0;
}

int PerfTimer::getLevel() const
{
	return level_;
}

double PerfTimer::getFPS() const
{
	return getCount() / getTime();
}


// ====================================================


CodeTimer::CodeTimer(void)
{
	counter_ = 0;
	cur_tree_level_ = 0;
}


CodeTimer::~CodeTimer(void)
{
}

void CodeTimer::start( const string& timerName, int level, char *timerDescriptor /*= NULL*/ )
{
	if(timer_pool_.find(timerName) == timer_pool_.end())
	{
		PerfTimer timer(timerName, level, timerDescriptor);
		timer_pool_[timerName] = TimerNode(timer, counter_, cur_tree_level_);//timer;
		counter_++;
	}
	cur_tree_level_++;
	timer_pool_[timerName].timer.start();
}

void CodeTimer::stop( const string& timerName )
{
//	assert(timer_pool_.find(timerName) != timer_pool_.end());
	if(timer_pool_.find(timerName) == timer_pool_.end())
	{
		printf("Timer %s not completed\.\n", timerName.c_str());
		return;
	}
	timer_pool_[timerName].timer.stop();
	cur_tree_level_--;
}

void CodeTimer::printTimes(int minPrintLevel, bool isShowAverage)
{
	for(int i = 0; i < counter_; i++)
	{
		TimerMap::iterator iter = timer_pool_.begin();
		while(iter != timer_pool_.end())
		{
			if(iter->second.count == i)
			{
				int lv = iter->second.timer.getLevel();
				if(lv >= minPrintLevel)
				{
					string str = iter->second.timer.getFormattedTime(isShowAverage);
					printf("%s\n", str.c_str());
				}
				break;
			}
			iter++;
		}
	}
}

double total_time = 0;
double total_count = 0;
double total_fps = 0;

void CodeTimer::printTimeTree( int minPrintLevel /*= 0*/, bool isShowAverage /*= false*/ )
{
	for(int i = 0; i < counter_; i++)
	{
		TimerMap::iterator iter = timer_pool_.begin();
		while(iter != timer_pool_.end())
		{
			if(iter->second.count == i)
			{
				int lv = iter->second.timer.getLevel();
				if(lv >= minPrintLevel)
				{
					/*string str = iter->second.timer.getFormattedTime(isShowAverage);
					for(int kk = 0; kk < iter->second.treeLevel; kk++)
					{
						printf("  ");
					}
					printf("%s\n", str.c_str());*/
					total_fps = iter->second.timer.getAverageTime() * 1000;
				}
				break;
			}
			iter++;
		}
	}
}

void CodeTimer::clear()
{
	timer_pool_.clear();
}

// for global use
CodeTimer gCodeTimer;
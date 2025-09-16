#include "stm32f10x.h"
#include "misc.h"
#include "delay.h"
#include "ALL_DATA.h"
static volatile uint32_t usTicks = 0;


/**
  * @brief  This function handles SysTick_Handler exception.
  * @param  None
  * @retval None
  */
void SysTick_Handler(void)
{
	SysTick_count++;
}



void cycleCounterInit(void)
{
	//滴答定时器初始化
	SysTick_Config(SystemCoreClock / 1000);	
}

/***********************************************************************
 * 获得系统当前运行的us值 
 * @param[in] 
 * @param[out] 
 * @return     
 **********************************************************************/
uint32_t GetSysTime_us(void) //获取当前系统的运行时间 单位us
{
    return (SysTick_count*1000  + (72000 - SysTick->VAL) / 72); //SysTick->VAL 当前倒计数的值//72000每个ms计数的重装值//1000 ms转化成us
}

//    毫秒级延时函数	 
void delay_ms(uint16_t nms)
{
	uint32_t t0=GetSysTime_us();
	while(GetSysTime_us() - t0 < nms * 1000);	  	  
}

void delay_us(unsigned int i)
 {  
	char x=0;   
    while( i--)
    {	
       for(x=1;x>0;x--);
    }
 }		  

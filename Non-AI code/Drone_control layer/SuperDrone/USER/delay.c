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
	//�δ�ʱ����ʼ��
	SysTick_Config(SystemCoreClock / 1000);	
}

/***********************************************************************
 * ���ϵͳ��ǰ���е�usֵ 
 * @param[in] 
 * @param[out] 
 * @return     
 **********************************************************************/
uint32_t GetSysTime_us(void) //��ȡ��ǰϵͳ������ʱ�� ��λus
{
    return (SysTick_count*1000  + (72000 - SysTick->VAL) / 72); //SysTick->VAL ��ǰ��������ֵ//72000ÿ��ms��������װֵ//1000 msת����us
}

//    ���뼶��ʱ����	 
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

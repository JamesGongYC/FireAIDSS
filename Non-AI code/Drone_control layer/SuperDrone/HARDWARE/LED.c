#include "stm32f10x.h"
#include "LED.h"
#include "ALL_DATA.h"





////右前灯			 
#define fLED1_H()  TIM3->CCR1=1000 //暗
#define fLED1_L()  TIM3->CCR1=500  //亮
#define fLED1_Toggle()  TIM3->CCR1^=(1000^500)//闪烁
////左前灯			 
#define fLED2_H()  TIM3->CCR2=1000 //暗
#define fLED2_L()  TIM3->CCR2=500  //亮
#define fLED2_Toggle()  TIM3->CCR2^=(1000^500)//闪烁
////左后灯			 
#define bLED3_H()  TIM3->CCR3=1000 //暗
#define bLED3_L()  TIM3->CCR3=500  //亮
#define bLED3_Toggle()  TIM3->CCR3^=(1000^500)//闪烁
//-------------------------------------------------
////左后灯			 
#define bLED4_H()  TIM3->CCR4=1000 //暗
#define bLED4_L()  TIM3->CCR4=500  //亮
#define bLED4_Toggle()  TIM3->CCR4^=(1000^500)//闪烁
//-------------------------------------------------

//-------------------------------------------------
//---------------------------------------------------------
/*     you can select the LED statue on enum contains            */
sLED LED = {300,AllFlashLight};  //LED initial statue is off;
                             //default 300ms flash the status

/**************************************************************
 *  LED system
 * @param[in] 
 * @param[out] 
 * @return     
 ***************************************************************/	
void PilotLED() //flash 300MS interval
{
	static uint32_t LastTime = 0;
//	static u8 last_LED_status
	if(SysTick_count - LastTime < LED.FlashTime)
	{

		return;
	}
	else
		LastTime = SysTick_count;
	if(!ALL_flag.unlock)//如果锁定状态无论如何4个LED都是闪烁
	{
		LED.status = AllFlashLight;	
	}
	else
	{
		if(!ALL_flag.flow_control&&!ALL_flag.height_lock)//如果不是定高也不是定点状态，则为4个LED常亮
		{
			LED.status = AlwaysOn;	
		}
		else if(ALL_flag.flow_control)//定点状态，4个灯全暗
		{
			LED.status = AlwaysOff;	
		}
		else
		{
			LED.status = WARNING;	//定高状态，机头两个灯闪烁	
		}
	}
	
	
	switch(LED.status)
	{
		case AlwaysOff:      //常暗   
			fLED1_H();
			fLED2_H();
			bLED3_H();
			bLED4_H();
			break;
		case AllFlashLight:				  //全部同时闪烁		
			fLED1_Toggle();
			fLED2_Toggle();	
			bLED3_Toggle();  
			bLED4_Toggle();  
	
		  break;
		case AlwaysOn:  //常亮
			fLED1_L();
			fLED2_L();
			bLED3_L();
			bLED4_L();

		  break;
		case WARNING://前面两灯闪烁
			fLED1_Toggle();		
			fLED2_Toggle();	
			bLED3_H();
			bLED4_H();
			break;
		default:
			LED.status = AlwaysOff;
			break;
	}
}

/**************************END OF FILE*********************************/




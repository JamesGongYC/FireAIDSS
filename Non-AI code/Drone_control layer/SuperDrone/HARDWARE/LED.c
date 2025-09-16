#include "stm32f10x.h"
#include "LED.h"
#include "ALL_DATA.h"





////��ǰ��			 
#define fLED1_H()  TIM3->CCR1=1000 //��
#define fLED1_L()  TIM3->CCR1=500  //��
#define fLED1_Toggle()  TIM3->CCR1^=(1000^500)//��˸
////��ǰ��			 
#define fLED2_H()  TIM3->CCR2=1000 //��
#define fLED2_L()  TIM3->CCR2=500  //��
#define fLED2_Toggle()  TIM3->CCR2^=(1000^500)//��˸
////����			 
#define bLED3_H()  TIM3->CCR3=1000 //��
#define bLED3_L()  TIM3->CCR3=500  //��
#define bLED3_Toggle()  TIM3->CCR3^=(1000^500)//��˸
//-------------------------------------------------
////����			 
#define bLED4_H()  TIM3->CCR4=1000 //��
#define bLED4_L()  TIM3->CCR4=500  //��
#define bLED4_Toggle()  TIM3->CCR4^=(1000^500)//��˸
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
	if(!ALL_flag.unlock)//�������״̬�������4��LED������˸
	{
		LED.status = AllFlashLight;	
	}
	else
	{
		if(!ALL_flag.flow_control&&!ALL_flag.height_lock)//������Ƕ���Ҳ���Ƕ���״̬����Ϊ4��LED����
		{
			LED.status = AlwaysOn;	
		}
		else if(ALL_flag.flow_control)//����״̬��4����ȫ��
		{
			LED.status = AlwaysOff;	
		}
		else
		{
			LED.status = WARNING;	//����״̬����ͷ��������˸	
		}
	}
	
	
	switch(LED.status)
	{
		case AlwaysOff:      //����   
			fLED1_H();
			fLED2_H();
			bLED3_H();
			bLED4_H();
			break;
		case AllFlashLight:				  //ȫ��ͬʱ��˸		
			fLED1_Toggle();
			fLED2_Toggle();	
			bLED3_Toggle();  
			bLED4_Toggle();  
	
		  break;
		case AlwaysOn:  //����
			fLED1_L();
			fLED2_L();
			bLED3_L();
			bLED4_L();

		  break;
		case WARNING://ǰ��������˸
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




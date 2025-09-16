#include "ALL_DEFINE.h"
#include "scheduler.h"
#include "flow.h"

volatile uint16_t cnt_600ms=0; 
void TIM1_UP_IRQHandler(void)   //TIM3�ж� 3ms
{
	
	if (TIM_GetITStatus(TIM1, TIM_IT_Update) != RESET) //���ָ����TIM�жϷ������:TIM �ж�Դ 
	{
		static uint8_t cnt_3ms = 0;
		static uint8_t cnt_6ms = 0;	 
		static uint8_t cnt_24ms = 0;			
		SysTick_count++;
		cnt_3ms++;
		cnt_6ms++;
		cnt_24ms++;	
		
		if(ALL_flag.usart_control)
		{
			cnt_600ms++;
		}else{
			cnt_600ms=0;
		}
		
		if(cnt_24ms>=8)
		{
				cnt_24ms = 0;				
				flow_data_sins();
				FlowPidControl(0.024);
		}		
		
		if(cnt_3ms == 1)  //3ms����һ��
		{
			// if(ALL_flag.flow_control==0)ALL_flag.usart_control=0;//������ǹ����رմ���
			cnt_3ms = 0;
			MpuGetData();
			RC_Analy();			
			FlightPidControl(0.003f);
			MotorControl();
		}
		
		if(cnt_6ms == 2) //6ms����һ��
		{
			cnt_6ms = 0;
			GetAngle(&MPU6050,&Angle,0.00626f);
			HeightPidControl(0.006f); //�߶ȿ���
			flow_get_data();
		}
		if(ALL_flag.usart_control&&cnt_600ms>=50) //600ms
		{        
			cnt_600ms=50;
			Remote.roll=1500;
			Remote.pitch=1500;
			Remote.thr=1500;
			Remote.yaw=1500; 
		}	

		TIM_ClearITPendingBit(TIM1,TIM_IT_Update );  //���TIMx���жϴ�����λ:TIM �ж�Դ 
	}
	
}



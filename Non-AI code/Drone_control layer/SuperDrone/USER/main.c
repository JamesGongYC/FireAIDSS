#include "ALL_DEFINE.h"

//�ر��������������˿տ��ĵش��������ڽ��з��С������ǳ�������������ɽ����ر�ң�ء�

//           ��ͷ      
//   PWM3     ��       PWM1
//      *           *
//      	*       *
//    		  *   *
//      			*  
//    		  *   *
//      	*       *
//      *           *
//    PWM4           PWM2			
int main(void)
{	
	cycleCounterInit();  //�õ�ϵͳÿ��us��ϵͳCLK������Ϊ�Ժ���ʱ�������͵õ���׼�ĵ�ǰִ��ʱ��ʹ��
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2); //2��bit����ռ���ȼ���2��bit�������ȼ�
	SysTick_Config(SystemCoreClock / 1000);	//ϵͳ�δ�ʱ��

	ALL_Init();//ϵͳ��ʼ��
	
	/*
	 whileѭ������һЩ�Ƚϲ���Ҫ�����飬������λ����LED���ơ�
	���๦�ܽ԰������жϣ���������ֻ��һ��3ms����һ�ε��жϣ����scheduler.c����Ĺ���
	*/
	while(1)
	{
			ANTO_polling(); //������λ����������
			PilotLED(); //LEDˢ��
	}
}


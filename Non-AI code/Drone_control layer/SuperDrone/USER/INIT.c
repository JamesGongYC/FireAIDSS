#include "ALL_DEFINE.h"


volatile uint32_t SysTick_count; //ϵͳʱ�����
_st_Mpu MPU6050;   //MPU6050ԭʼ����
_st_AngE Angle;    //��ǰ�Ƕ���ֵ̬
_st_Remote Remote; //ң��ͨ��ֵ


_st_ALL_flag ALL_flag; //ϵͳ��־λ������������־λ��


PidObject pidRateX; //�ڻ�PID����
PidObject pidRateY;
PidObject pidRateZ;

PidObject pidPitch; //�⻷PID����
PidObject pidRoll;
PidObject pidYaw;


PidObject pidHeightRate;
PidObject pidHeightHigh;


PidObject pidPositionX;
PidObject pidPositionY;

PidObject pidPositionX;
PidObject pidPositionY;
PidObject pidPosRateX;
PidObject pidPosRateY;

void pid_param_Init(void); //PID���Ʋ�����ʼ������дPID�����ᱣ�����ݣ��������ɺ�ֱ���ڳ�������� ����¼���ɿ�


int16_t motor_PWM_Value[4];//


void ALL_Init(void)
{
	delay_ms(1000);    //����ʹ��USB������Ҫ��ʱ���¸ղ����ѹ���ȶ�
	
	USB_HID_Init();   		//USB��λ����ʼ��
	
	IIC_Init();             //I2C��ʼ��
	
	pid_param_Init();       //PID������ʼ��
	
	MpuInit();              //MPU6050��ʼ��
//----------------------------------------	
// ˮƽ��ֹ�궨���ù���ֻ��Ҫ����һ�Σ���Ҫÿ�ν��С���ҷ���ǰ�Ѿ�����һ�α궨�ˣ��궨�����Զ����浽MCU��FLASH�С�
// ����У׼�����´򿪼��ɣ���ʱ5S��Ϊ�˲��ϵ�غ��г����ʱ�佫���������ڵ��Ͻ���ˮƽ��ֹ�궨��
//delay_ms(1000); 	MpuGetOffset();
//----------------------------------------		
	USART1_Config();  //���ô���     

	
	NRF24L01_init();				//2.4Gң��ͨ�ų�ʼ��
	
	TIM2_PWM_Config();			//4·PWM��ʼ��
	TIM3_PWM_Config();      //LED PWM��ʼ��
	TIM1_Config();					//ϵͳ�������ڳ�ʼ�� 
	

	
}


//PID�ڴ˴��޸�
void pid_param_Init(void)//PID������ʼ��
{
	pidRateX.kp = 3.f;
	pidRateY.kp = 3.f;
	pidRateZ.kp = 8.0f;
	
	
	pidRateX.kd = 0.2;
	pidRateY.kd = 0.2;
	pidRateZ.kd = 0.4f;	
	
	pidPitch.kp = 8.0f;
	pidRoll.kp = 8.0f;
	pidYaw.kp = 6.0f;	
	





		//�ڻ�PID���� �ٶ�
	pidHeightRate.kp = 1.2f; //1.2f
	pidHeightRate.ki = 0.04f;
	pidHeightRate.kd = 0.085f;
		//�⻷PID����
	pidHeightHigh.kp = 1.25f;//1.2f
	pidHeightHigh.ki = 0.00f;
	pidHeightHigh.kd = 0.085f;//0.085f
	
	
//		pidPosRateX.kp = 0.25f;
//	pidPosRateY.kp = 0.25f;
//	pidPosRateX.kd = 0.035f;
//	pidPosRateY.kd = 0.035f;	
//	pidPosRateX.ki = 0.04f;
//	pidPosRateY.ki = 0.04f;
//	
//	pidPositionX.kp = 0.02f;
//	pidPositionY.kp  = 0.02f;


	pidPosRateX.kp = 0.15f;
	pidPosRateY.kp = 0.15f;
	pidPosRateX.kd = 0.035f;
	pidPosRateY.kd = 0.035f;	
	pidPosRateX.ki = 0.04f;
	pidPosRateY.ki = 0.04f;
	
	pidPositionX.kp = 0.02f;
	pidPositionY.kp  = 0.02f;




}












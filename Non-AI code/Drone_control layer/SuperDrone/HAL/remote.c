#include "ALL_DATA.h"
#include "nrf24l01.h"
#include "control.h"
#include <math.h>
#include "myMath.h"
#include "LED.h"
#include "Remote.h"
#include "mpu6050.h"
#include "stdio.h"

#define SUCCESS 0
#undef FAILED
#define FAILED  1
/*****************************************************************************************
 *  ͨ�����ݴ���
 * @param[in] 
 * @param[out] 
 * @return     
 ******************************************************************************************/	
	uint8_t RC_rxData[32];
void remote_unlock(void);
uint16_t last_Remote_AUX1 = 1000;
uint16_t last_Remote_AUX3 = 1000;
uint16_t last_Remote_AUX4 = 1000;
uint16_t last_Remote_AUX2 = 1000;

int16_t roll_offset; //΢��ֵ
int16_t pitch_offset;


void RC_Analy(void)  
{
	static uint16_t cnt;
	volatile static uint8_t status=WAITING_1; 
/*             Receive  and check RC data                               */	
	if(NRF24L01_RxPacket(RC_rxData)==SUCCESS)
	{ 	

		uint8_t i;
		uint8_t CheckSum=0;
		cnt = 0;
		for(i=0;i<31;i++)
		{
			CheckSum +=  RC_rxData[i];
		}
		if(RC_rxData[31]==CheckSum && RC_rxData[0]==0xAA && RC_rxData[1]==0xAF)  //������յ���ң��������ȷ
		{

			  Remote.roll = ((uint16_t)RC_rxData[4]<<8) | RC_rxData[5];  //ͨ��1
				Remote.roll = LIMIT(Remote.roll,1000,2000);
				Remote.pitch = ((uint16_t)RC_rxData[6]<<8) | RC_rxData[7];  //ͨ��2
				Remote.pitch = LIMIT(Remote.pitch,1000,2000);

				if(ALL_flag.unlock!=1)
				{
					Remote.thr = 	((uint16_t)RC_rxData[8]<<8) | RC_rxData[9];   //ͨ��3
					Remote.thr = 	LIMIT(Remote.thr,1000,2000);
				}
				Remote.yaw =  ((uint16_t)RC_rxData[10]<<8) | RC_rxData[11];   //ͨ��4
				Remote.yaw =  LIMIT(Remote.yaw,1000,2000);
				Remote.AUX1 =  ((uint16_t)RC_rxData[12]<<8) | RC_rxData[13];   //ͨ��5  ���Ͻǰ���������ͨ��5  
				Remote.AUX1 =  LIMIT(Remote.AUX1,1000,2000);
				Remote.AUX2 =  ((uint16_t)RC_rxData[14]<<8) | RC_rxData[15];   //ͨ��6  ���Ͻǰ���������ͨ��6 
				Remote.AUX2 =  LIMIT(Remote.AUX2,1000,2000);
				Remote.AUX3 =  ((uint16_t)RC_rxData[16]<<8) | RC_rxData[17];   //ͨ��7  ���±߰���������ͨ��7 
				Remote.AUX3 =  LIMIT(Remote.AUX3,1000,2000);
				Remote.AUX4 =  ((uint16_t)RC_rxData[18]<<8) | RC_rxData[19];   //ͨ��8  ���±߰���������ͨ��6  
				Remote.AUX4 = LIMIT(Remote.AUX4,1000,4000);		
				
				if(value5==2)
				{
					if(ALL_flag.usart_control==1 && ALL_flag.flow_control==0 && ALL_flag.height_lock==0)
					{
						Remote.thr = 2000;
						// printf("2");
					}
				}
				else if(value5==3)
				{
					// if(ALL_flag.flow_control==0)ALL_flag.usart_control=0
					ALL_flag.flow_control=1;
					ALL_flag.height_lock=1;
					ALL_flag.usart_control=1;

					Remote.roll=1500;
					Remote.pitch=1500;
					Remote.thr=1500;
					Remote.yaw=1500;
					// printf("3");
				}
				if((Remote.AUX2!=last_Remote_AUX2)&&(ALL_flag.height_lock==1)&&(ALL_flag.flow_control==1))	
				{
					last_Remote_AUX2=Remote.AUX2;
					ALL_flag.usart_control^=1; 
					if(ALL_flag.usart_control)
					printf("���봮�ڿ���ģʽ\r\n");
					else 
					printf("�˳����ڿ���ģʽ\r\n");

					Remote.roll=1500;
					Remote.pitch=1500;
					Remote.thr=1500;
					Remote.yaw=1500; 
				}					 

				if(ALL_flag.usart_control&&ALL_flag.flow_control)
				{
					 Remote.roll=value1; //��һ������
					 Remote.pitch=value2;//�ڶ�������
					 Remote.thr=value3;
					 Remote.yaw=value4; 
					//  const float yaw_ratio =   0.5f;		//ƫ���������ȶ�

					if(value5==0)
					{
						// printf("000");
						if(Remote.yaw>1820)
						{
							pidYaw.desired -= 0.5f;	//���ſ���ƫ���Ƿ���
						}
						else if(Remote.yaw <1180)
						{
							pidYaw.desired += 0.5f;	//���ſ���ƫ���Ƿ���
						}		

						if (Remote.thr>1820)
						{
							pidHeightHigh.desired -= 0.1f;
							if(pidHeightHigh.desired >= 2000)
							{
								pidHeightHigh.desired = 2000;
							}
						}
						else if(Remote.thr<1180)
						{
							pidHeightHigh.desired += 0.1f;
							if(pidHeightHigh.desired <= 1000)
							{
								pidHeightHigh.desired = 1000;
							}
						}
					}
					else if(value5==1)
					{
						// printf("111");
						Remote.thr = 1000;
						Remote.yaw = 1000;
						ALL_flag.flow_control = 0;
						ALL_flag.height_lock = 0;
						last_Remote_AUX1 = Remote.AUX1;
						last_Remote_AUX2 = Remote.AUX2;
					}
						
				
				 }

							//ͨ��5��Ϊ���߿��� ÿ��һ�¾ͻ���1000��2000֮���л� 
							if(Remote.AUX1!=last_Remote_AUX1) //�������1700 ����붨��ģʽ
							{
								last_Remote_AUX1=Remote.AUX1;
								
								Remote.AUX2=last_Remote_AUX2=1000;
								ALL_flag.height_lock = ALL_flag.flow_control ^= 1;//ȡ��
								if(ALL_flag.flow_control)
								printf("���붨�߿���ģʽ\r\n");
								else
								printf("�˳����߿���ģʽ\r\n");
								ALL_flag.usart_control=0;
							}														
						
							{
								if(!ALL_flag.flow_control)//��������ڶ���ģʽ����нǶ���ң�ص�ң�˿���
								{
											float roll_pitch_ratio = 0.04f;
											pidPitch.desired =(-(Remote.pitch-1500)+pitch_offset)*roll_pitch_ratio;	 //��ң��ֵ��Ϊ���нǶȵ�����ֵ
											pidRoll.desired = (-(Remote.roll-1500)+roll_offset)*roll_pitch_ratio;	
								}

								{
										const float yaw_ratio =   0.5f;		//ƫ���������ȶ�
								
										if(Remote.yaw>1820)
										{
											pidYaw.desired -= yaw_ratio;	//���ſ���ƫ���Ƿ���
										}
										else if(Remote.yaw <1180)
										{
											pidYaw.desired += yaw_ratio;	//���ſ���ƫ���Ƿ���
										}		
								}							
							}
							remote_unlock();
			
		}
  }
//���3��û�յ�ң�����ݣ����ж�ң���źŶ�ʧ���ɿ����κ�ʱ��ֹͣ���У��������ˡ�
//���������ʹ���߿ɽ����ر�ң�ص�Դ������������3��������رգ��������ˡ�
//�����ر�ң�أ�����ڷ����л�ֱ�ӵ��䣬���ܻ��𻵷�������
	else
	{
	
		
		cnt++;
		if(cnt>500)
		{
			cnt = 0;
			ALL_flag.unlock = 0; 
			NRF24L01_init();
		}
	}
}

/*****************************************************************************************
 *  �����ж�
 * @param[in] 
 * @param[out] 
 * @return     
 ******************************************************************************************/	
void remote_unlock(void)
{
	volatile static uint8_t status=WAITING_1;
	static uint16_t cnt=0;

	if(Remote.thr<1050 &&Remote.yaw<1150)                         //����ң�����½�����
	{
		status = EXIT_255;
	}
	
	switch(status)
	{
		case WAITING_1://�ȴ�����
			if(Remote.thr<1150)           //���������࣬�������->�������->������� ����LED�Ʋ����� ����ɽ���
			{			 
					 status = WAITING_2;				 
			}		
			break;
		case WAITING_2:
			if(Remote.thr>1800)          
			{		
						static uint8_t cnt = 0;
					 	cnt++;		
						if(cnt>5) //��������豣��200ms���ϣ���ֹң�ؿ�����ʼ��δ��ɵĴ�������
						{	
								cnt=0;
								status = WAITING_3;
						}
			}			
			break;
		case WAITING_3:
			if(Remote.thr<1100)          
			{			 
					 status = WAITING_4;				 
			}			
			break;			
		case WAITING_4:	//����ǰ׼��	               
				ALL_flag.unlock = 1;
				status = PROCESS_31;
				LED.status = AlwaysOn;	
				last_Remote_AUX1=Remote.AUX1;
				last_Remote_AUX2=Remote.AUX2;

				ALL_flag.usart_control = 1;
				// printf("www");
				ALL_flag.height_lock = 0;//��������̬ģʽ�����
				ALL_flag.flow_control = 0;//��������̬ģʽ�����	
				pidRoll.offset =   -(Remote.roll-1500)*0.04;//΢��ֵ��ǰ��
				pidPitch.offset  = -(Remote.pitch-1500)*0.04;
				roll_offset = (Remote.roll-1500); //��¼΢��ֵ
				pitch_offset = (Remote.pitch-1500);
				 break;		
		
		case PROCESS_31:	//�������״̬
				if(Remote.thr<1020)
				{
					if(!ALL_flag.height_lock )
					{
						if(cnt++ > 3000)                                     // ����ң�˴������9S�Զ�����
						{								
							status = EXIT_255;								
						}
					}
				}
				else if(!ALL_flag.unlock)                           //Other conditions lock 
				{
					status = EXIT_255;				
				}
				else					
					cnt = 0;
			break;
		case EXIT_255: //��������
			LED.status = AllFlashLight;	                                 //exit
			cnt = 0;
			LED.FlashTime = 100; //100*3ms		
			ALL_flag.unlock = 0;
			status = WAITING_1;
			break;
		default:
			status = EXIT_255;
			break;
	}
}
/***********************END OF FILE*************************************/








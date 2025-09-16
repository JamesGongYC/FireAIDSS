#include "ALL_DATA.h" 
#include "ALL_DEFINE.h" 
#include "control.h"
#include "pid.h"
#include "flow.h"
#include "kalman.h"

//------------------------------------------------------------------------------
#undef NULL
#define NULL 0
#undef DISABLE 
#define DISABLE 0
#undef ENABLE 
#define ENABLE 1
#undef REST
#define REST 0
#undef SET 
#define SET 1 
#undef EMERGENT
#define EMERGENT 0

//------------------------------------------------------------------------------
// �ṹ�����飬��ÿһ�������һ��pid�ṹ�壬�����Ϳ���������������PID��������  �������ʱ������λpid�������ݣ�����������仰�����þͿ�����
PidObject *(pPidObject[])={&pidRateX,&pidRateY,&pidRateZ,&pidRoll,&pidPitch,&pidYaw   
		,&pidHeightRate
		,&pidHeightHigh
		,&pidPosRateX
		,&pidPosRateY		
		,&pidPositionX
		,&pidPositionY	
};


float sins_high;
float sins_vel;

uint16_t thr_hold = 0; //����߶�ʱ��¼��ǰ����ֵ
/**************************************************************
 *  Height control  //������װLC06�߶ȼƵ����ʹ�ã����������˺��������̷���
 * @param[in] 
 * @param[out] 
 * @return     
 ***************************************************************/
float last_high,last_vel;
 uint32_t VL53L01_high = 0; //��ǰ�߶�
static uint8_t high_error_count;
  uint8_t set_high = 0;
void HeightPidControl(float dt)
{
	volatile static uint8_t status=WAITING_1;
		int16_t acc;       //��ǰ��������ļ��ٶ�ֵ
   	int16_t acc_error; //��ǰ���ٶȼ�ȥ�������ٶ���Ϊ�����ƶ��ļ��ٶ�
   static int16_t acc_offset;//�������ٶ�ֵ
	 
	{ //��ȡ��ֱ�ٶ�����

		acc = (int16_t)GetNormAccz();//��ȡ��������
		
			if(!ALL_flag.unlock) //ȡ�þ�̬���ٶ�ֵ
			{
				acc_offset = acc;
			}
			acc_error = acc - acc_offset;	
			
			if(VL53L01_high<3500)			
			{//�˴���һ���ٶ���߶ȵĻ����˲� 
					if(VL53L01_high-sins_high>50)sins_high += 50;//�߶��쳣ͻ��
					else if(VL53L01_high-sins_high<-50)sins_high -= 50;//�߶��쳣ͻ��
					else 	sins_high= VL53L01_high;
				
					sins_vel=(last_vel + acc_error * dt)*0.985f+0.015f*(sins_high - last_high)/dt; //�ٶȻ���Զ�����������Ի����˲��ؼ���ץ���ٶȻ�����
			
				//sins_high= high;
				pidHeightRate.measured = last_vel=	sins_vel;
				pidHeightHigh.measured=last_high = sins_high;  //�߶ȸ���

			}	
		}
	//----------------------------------------------������ֹ����
	if(ALL_flag.unlock == EMERGENT) //�����������ʹ��ң�ؽ����������ɿؾͿ������κ�����½�����ֹ���У��������������˳�PID����
		status = EXIT_255;
	//----------------------------------------------����
	switch(status)
	{
		case WAITING_1: 	// ������
		  if( ALL_flag.unlock) 
			{
				pidHeightRate.measured=0;
				
				pidHeightRate.measured = last_vel=	sins_vel=0;
				pidHeightHigh.measured=last_high = sins_high;
				status = WAITING_2;
				high_error_count=0;
			}
			break;
		case WAITING_2: 	// ��ⶨ�ߣ�����ǰ׼��
			if(ALL_flag.height_lock) 
			{
				
				set_high=0;

				LED.status = WARNING;
			thr_hold=500;
			status = PROCESS_31;
			}
			break;
		
		case PROCESS_31:	// ���붨��	
		
			 if(Remote.thr<1750 && Remote.thr>1150) // ��������ѻ��У������߶�
			 {
						if(set_high == 0) 								// ������˳����ߣ���¼��ǰλ��Ϊ����λ��
						{
							set_high = 1;
							pidHeightHigh.desired = pidHeightHigh.measured;//��¼���Ż��еĸ߶ȵ�����ǰ���߸߶�
						}
						pidUpdate(&pidHeightHigh,dt);    	// ����PID���������������⻷	������PID	
						pidHeightRate.desired = pidHeightHigh.out;  
			 }
			else if(Remote.thr>1750) 								// �������������� �����߶�
			{
				if(VL53L01_high<3500)									// �߶ȴ���3500mm̫�߲�Ҫ��
				{
					set_high = 0;
					pidHeightRate.desired = 250; 				// �����ٶȿɵ�
				}
			}
			else if	(Remote.thr<1150) 							// �����������½�	�����߶�	
			{
				set_high = 0;
				pidHeightRate.desired = -350; 				// �½��ٶȿɵ�
				if(pidHeightHigh.measured<10)					// �ŵ�
				{
					ALL_flag.unlock = 0;
				}
			}					 
								 
			pidUpdate(&pidHeightRate,dt); 	// �ٵ����ڻ�				
				 
			if(!ALL_flag.height_lock)  			// �˳�����,��ң�ز����˳����߻��ߴ���3.5m�������������
			{
				LED.status = AlwaysOn ;
				status = EXIT_255;
			}
			if(VL53L01_high<200||VL53L01_high>3700)	// �߶��쳣
			{
				high_error_count++;
				if(high_error_count>50)								// 50*6ms,300ms�����쳣
				{
						ALL_flag.height_lock=0;
						ALL_flag.flow_control=0;					// �˳����߶���
						status = EXIT_255;
				}
			}
			else
			{
				high_error_count=0;
			}
			break;
		case EXIT_255: 										// �˳�����
			pidRest(&pPidObject[6],2);			// �����ǰ�Ķ������ֵ
			status = WAITING_1;							// �ص��ȴ����붨��
			break;
		default:
			status = WAITING_1;
			break;	
	}	
}


/**************************************************************
 *  flow control
 * @param[in] 
 * @param[out] 
 * @return     
 ***************************************************************/
void FlowPidControl(float dt)
{
	volatile static uint8_t status=WAITING_1;
	static uint8_t set_pos = 0;

	//----------------------------------------------������ֹ����
	if(ALL_flag.unlock == EMERGENT) 		// �����������ʹ��ң�ؽ����������ɿؾͿ������κ�����½�����ֹ���У��������������˳�PID����
		status = EXIT_255;
	//----------------------------------------------����
	switch(status)
	{
		case WAITING_1:		
			if(ALL_flag.unlock) //������
			{
					status = WAITING_2;	
			}
			break;
		case WAITING_2: 								// �����Ŵ���1300  ��Ĭ��Ϊ��ɺ���붨�����
			if(ALL_flag.flow_control&&VL53L01_high>200) 
			{
				pidRest(&pPidObject[8],4);  // ��λ�ϴ�����������PID����
				status = PROCESS_31;
				set_pos = 1;
			}			
			break;
		case PROCESS_31:								// ����
			{
				if(!ALL_flag.usart_control)
				{
					if(Remote.roll>1750||Remote.roll<1250||Remote.pitch>1750||Remote.pitch<1250)
					{
							set_pos = 1;
							if(Remote.roll>1750)
								pidPosRateX.desired = -25;   	// ֱ�ӿ����ٶ� ���ҹص��⻷����
							else if(Remote.roll<1250)
								pidPosRateX.desired = 25; 
							if(Remote.pitch>1750)
								pidPosRateY.desired = 25;   	// ֱ�ӿ����ٶ� ���ҹص��⻷����
							else if(Remote.pitch<1250)
								pidPosRateY.desired = -25; 
					}
					else
					{	
						
						if(set_pos == 1)   		// ��¼λ�� һ��
						{
							set_pos = 0; 				// �رռ�¼λ�ñ�־

							pidPositionX.desired = flow_x_lpf_att_i;
							pidPositionY.desired = flow_y_lpf_att_i;	
						}
						
						// �⻷λ�ÿ���
						pidPositionX.measured = flow_x_lpf_att_i;	// ʵʱλ�÷���
						pidUpdate(&pidPositionX,dt);							// λ������PID
						pidPositionY.measured = flow_y_lpf_att_i;	// ʵʱλ�÷���
						pidUpdate(&pidPositionY,dt);							// λ������PID
						// �ڻ�����
						pidPosRateX.desired = LIMIT(pidPositionX.out,-20,20);	// λ��PID������ٶ�����
						pidPosRateY.desired = LIMIT(pidPositionY.out,-20,20);	// λ��PID������ٶ�����
					}
				}
				if(ALL_flag.usart_control)
				{
					if(Remote.roll>1550||Remote.roll<1450||Remote.pitch>1550||Remote.pitch<1450)
					{
						set_pos = 1;
						pidPosRateX.desired=(1500-Remote.roll)*0.03;
					  pidPosRateY.desired=(1500-Remote.pitch)*0.03;
//						if(Remote.roll>1650)
//							pidPosRateX.desired = -20;   		// ֱ�ӿ����ٶ� ���ҹص��⻷����
//						else if(Remote.roll<1250)
//							pidPosRateX.desired = 20; 
//						if(Remote.pitch>1650)
//							pidPosRateY.desired = 20;   		// ֱ�ӿ����ٶ� ���ҹص��⻷����
//						else if(Remote.pitch<1350)
//							pidPosRateY.desired = -20;
					}
					else
					{	
						if(set_pos == 1)   	// ��¼λ�� һ��
						{
							set_pos = 0; 			// �رռ�¼λ�ñ�־

							pidPositionX.desired = flow_x_lpf_att_i;
							pidPositionY.desired = flow_y_lpf_att_i;	
						}
						
						// �⻷λ�ÿ���
						pidPositionX.measured = flow_x_lpf_att_i;		// ʵʱλ�÷���
						pidUpdate(&pidPositionX,dt);								// λ������PID
						pidPositionY.measured = flow_y_lpf_att_i;		// ʵʱλ�÷���
						pidUpdate(&pidPositionY,dt);								// λ������PID
						// �ڻ�����
						pidPosRateX.desired = LIMIT(pidPositionX.out,-30,30);	// λ��PID������ٶ�����
						pidPosRateY.desired = LIMIT(pidPositionY.out,-30,30);	// λ��PID������ٶ�����
					}
				}
				//�ڻ�
				pidPosRateX.measured = flow_x_vel_lpf_i;//�ٶȷ���
				pidUpdate(&pidPosRateX,dt);//�ٶ�����
				pidPosRateY.measured = flow_y_vel_lpf_i;//�ٶȷ���
				pidUpdate(&pidPosRateY,dt);//�ٶ�����
				
				
				pidRoll.desired = LIMIT(pidPosRateX.out,-15,15)  ; //��̬�⻷����ֵ
				pidPitch.desired = -LIMIT(pidPosRateY.out,-15,15); //��̬�⻷����ֵ		

				if(!ALL_flag.flow_control||!ALL_flag.height_lock)
				{
			
					status = EXIT_255;
				}
			}
			break;	
		case EXIT_255: //�˳�����
			pidRest(&pPidObject[8],4);	
			status = WAITING_1;
			break;
		default:
			status = WAITING_1;
			break;		
	}
}


/**************************************************************
 *  flight control
 * @param[in] 
 * @param[out] 
 * @return     
 ***************************************************************/
void FlightPidControl(float dt)
{
	volatile static uint8_t status=WAITING_1;

	switch(status)
	{		
		case WAITING_1: //�ȴ�����
			if(ALL_flag.unlock)
			{
				status = READY_11;	
			}			
			break;
		case READY_11:  //׼���������
			pidRest(pPidObject,6); //������λPID���ݣ���ֹ�ϴ�����������Ӱ�챾�ο���

			Angle.yaw = pidYaw.desired =  pidYaw.measured = 0;   //����ƫ����
		
			status = PROCESS_31;
		
			break;			
		case PROCESS_31: //��ʽ�������
			if(Angle.pitch<-50||Angle.pitch>50||Angle.roll<-50||Angle.roll>50)//��б��⣬��Ƕ��ж�Ϊ������������������	
					if(Remote.thr>1200)//�����ŵĺܵ�ʱ������б��⣬��ֹ�ڿ�����������̫�ͣ����������������壬����������״̬δ֪
						ALL_flag.unlock = EMERGENT;//����������
					
      pidRateX.measured = MPU6050.gyroX * Gyro_G; //�ڻ�����ֵ �Ƕ�/��
			pidRateY.measured = MPU6050.gyroY * Gyro_G;
			pidRateZ.measured = MPU6050.gyroZ * Gyro_G;
		
			pidPitch.measured = Angle.pitch; //�⻷����ֵ ��λ���Ƕ�
		  pidRoll.measured = Angle.roll;
			pidYaw.measured = Angle.yaw;
		
		 	pidUpdate(&pidRoll,dt);    //����PID���������������⻷	�����PID		
			pidRateX.desired = pidRoll.out; //���⻷��PID�����Ϊ�ڻ�PID������ֵ��Ϊ����PID
			pidUpdate(&pidRateX,dt);  //�ٵ����ڻ�

		 	pidUpdate(&pidPitch,dt);    //����PID���������������⻷	������PID	
			pidRateY.desired = pidPitch.out;  
			pidUpdate(&pidRateY,dt); //�ٵ����ڻ�

			pidUpdate(&pidYaw,dt);    //����PID���������������⻷	ƫ����PID	
			pidRateZ.desired = pidYaw.out;  
			pidUpdate(&pidRateZ,dt); //�ٵ����ڻ�
			break;
		case EXIT_255:  //�˳�����
			pidRest(pPidObject,6);
			status = WAITING_1;//���صȴ�����
		  break;
		default:
			status = EXIT_255;
			break;
	}
	if(ALL_flag.unlock == EMERGENT) //�����������ʹ��ң�ؽ����������ɿؾͿ������κ�����½�����ֹ���У��������������˳�PID����
		status = EXIT_255;
}
/**************************************************************
 *  motor out
 * @param[in] 
 * @param[out] 
 * @return     
 ***************************************************************/

#define MOTOR1 motor_PWM_Value[0] 
#define MOTOR2 motor_PWM_Value[1] 
#define MOTOR3 motor_PWM_Value[2] 
#define MOTOR4 motor_PWM_Value[3] 
uint16_t low_thr_cnt;

void MotorControl(void)
{	
	volatile static uint8_t status=WAITING_1;
	
	
	if(ALL_flag.unlock == EMERGENT) //�����������ʹ��ң�ؽ����������ɿؾͿ������κ�����½�����ֹ���У��������������˳�PID����
		status = EXIT_255;	
	switch(status)
	{		
		case WAITING_1: //�ȴ�����	
			MOTOR1 = MOTOR2 = MOTOR3 = MOTOR4 = 0;  //������������������Ϊ0
			if(ALL_flag.unlock)
			{
				status = WAITING_2;
			}
		case WAITING_2: //������ɺ��ж�ʹ�����Ƿ�ʼ����ң�˽��з��п���
			if(Remote.thr>1100)
			{
				status = PROCESS_31;
			}
			break;
		case PROCESS_31:
			{
				int16_t thr;
				if(ALL_flag.height_lock) //����ģʽ�� ����ң����Ϊ�����߶�ʹ��
				{		
					thr = pidHeightRate.out+thr_hold; //�����������Ƕ������ֵ
				}
				else //��������״̬����������ʹ��
				{
					int16_t temp;
					temp = Remote.thr -1000; //����+�������ֵ					
						//���ű����滮
						thr = 200+0.4f * temp;
						thr_hold = thr;
						if(temp<10) 
						{
							
							low_thr_cnt++;
							if(low_thr_cnt>300)//1500ms
							{
								thr = 0;
								
								pidRest(pPidObject,6);
								MOTOR1 = MOTOR2 = MOTOR3 = MOTOR4 =0;
								status = WAITING_2;
								break;
							}
						}
						else low_thr_cnt=0;
				}
				
				//������ֵ��Ϊ����ֵ��PWM
				MOTOR1 = MOTOR2 = MOTOR3 = MOTOR4 = LIMIT(thr,0,800); //��200����̬����
//����������������ȡ���ڵ��PWM�ֲ���ɿ�������ϵ���뿴�ɿ�������ϵͼ�⣬���ĸ����PWM�ֲ��ֲ�	
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
//		pidRateX.out ����Ǵ���PID��� �������ң����Կ���1 2��3 4������������ͬ��ͬ��
//    pidRateY.out �����Ǵ���PID��� ����ǰ�󣬿��Կ���2 3��1 4��ǰ��������ͬ��ͬ��
//		pidRateZ.out ����Ǵ���PID��� ������ת�����Կ���2 4��1 3������Խ��ߵ��ͬ��ͬ��	

// ������ȡ�����㷨��� ������������Ļ�  ��ǰ�ɱ�Ȼ��β�������������,���ҷɱ�Ȼ����������������		

				MOTOR1 +=    + pidRateX.out + pidRateY.out + pidRateZ.out;//; ��̬����������������Ŀ�����
				MOTOR2 +=    + pidRateX.out - pidRateY.out - pidRateZ.out ;//;
				MOTOR3 +=    - pidRateX.out + pidRateY.out - pidRateZ.out;
				MOTOR4 +=    - pidRateX.out - pidRateY.out + pidRateZ.out;//;
			}	
			break;
		case EXIT_255:
			MOTOR1 = MOTOR2 = MOTOR3 = MOTOR4 = 0;  //������������������Ϊ0
			status = WAITING_1;	
			break;
		default:
			break;
	}
	
	
	TIM2->CCR1 = LIMIT(MOTOR1,0,1000);  //����PWM
	TIM2->CCR2 = LIMIT(MOTOR2,0,1000);
	TIM2->CCR3 = LIMIT(MOTOR3,0,1000);
	TIM2->CCR4 = LIMIT(MOTOR4,0,1000);
} 

/************************************END OF FILE********************************************/ 

#include "stm32f10x.h"
#include "ALL_DEFINE.h"
#include "flow.h"
#include "control.h"





 FLow flow_data;



//       ^ Y
//       |
//       |
//<------------X
//       |
//       |


float gyroX_i;
float gyroY_i;

float flow_x_i;
float flow_y_i;

float flow_x_vel_lpf_i;
float flow_y_vel_lpf_i;
float flow_x_pos_lpf_i;
float flow_y_pos_lpf_i;

float flow_x_lpf_att_i;
float flow_y_lpf_att_i;
float flow_x_att;
float flow_y_att;
float flow_vel_x_i;
float flow_vel_y_i;
float last_flow_pos_x_i;
float last_flow_pos_y_i;


#define  angle_to_rad  0.0174f  //�Ƕ�ת����


#include <math.h>
void flow_data_sins()
{

	if(flow_data.squal>30&&VL53L01_high>=80)
	{
//��ͨ�˲�
		flow_x_pos_lpf_i += (flow_data.pos_x_i - flow_x_pos_lpf_i) *0.10;
		flow_y_pos_lpf_i += (flow_data.pos_y_i - flow_y_pos_lpf_i) *0.10;
		//���λ�Ʋ�����������ͨ�˲�
		flow_x_att += (700.0f*tanf(Angle.roll *angle_to_rad) - flow_x_att) *0.10;
		flow_y_att += (700.0f*tanf(Angle.pitch*angle_to_rad) - flow_y_att) *0.10;
	
   //��ǲ������λ�����
	flow_x_lpf_att_i = (flow_x_pos_lpf_i+flow_x_att)*0.08f; 
	flow_y_lpf_att_i = (flow_y_pos_lpf_i-flow_y_att)*0.08f; 

	//��΢���ٶ�  
	flow_vel_x_i = (flow_x_lpf_att_i - last_flow_pos_x_i)*10;  
	flow_vel_y_i = (flow_y_lpf_att_i - last_flow_pos_y_i)*10;  	
  last_flow_pos_x_i = flow_x_lpf_att_i; 
	last_flow_pos_y_i = flow_y_lpf_att_i;  

	//��ͨ�˲�
	flow_x_vel_lpf_i += (  flow_vel_x_i - flow_x_vel_lpf_i ) * 0.15f;
	flow_y_vel_lpf_i += ( flow_vel_y_i - flow_y_vel_lpf_i ) * 0.15f;
	 //����߶Ȳ���λ��	//�߶�Խ�� λ����ֵӦ��Խ��
	flow_x_vel_lpf_i += +flow_vel_x_i*(VL53L01_high/3400.f);
	flow_y_vel_lpf_i += +flow_vel_y_i*(VL53L01_high/3400.f);
	}
	else 
	{
		flow_x_att=0;
		flow_y_att=0;
		flow_x_pos_lpf_i=0;
		flow_y_pos_lpf_i=0;
		flow_data.pos_x_i=0;
		flow_data.pos_y_i=0;
		flow_x_lpf_att_i=0;
  	flow_y_lpf_att_i=0;
		last_flow_pos_x_i=0;
		last_flow_pos_y_i=0;

		flow_x_vel_lpf_i=0;
		flow_y_vel_lpf_i=0;
		flow_vel_x_i=0;
		flow_vel_y_i=0;

	}
}



uint8_t temp_data[6];
uint8_t error_count=0;

void flow_get_data(void)
//----------------------------------------------	
{	//��ȡ��ǰ����߶�����  LC08�������ģ��

	#define LC08_ADDRESS 0xC4 //ģ���ַ ���λ��д��־λ 0��д  1����
	#define READ_ADDRESS 0X33 //�߶����ݼĴ�����ַ
	if(IIC_read_Bytes(LC08_ADDRESS,READ_ADDRESS,temp_data,5) != FAILED)//�����ȡ�ɹ�
	{
			u16 high;
			flow_data.vel_x =(*(int8_t*)&temp_data[2]);
			flow_data.vel_y = (*(int8_t*)&temp_data[3]);
			//�������ݻ��ֵó�λ��
			flow_data.pos_x_i += flow_data.vel_x;
			flow_data.pos_y_i += flow_data.vel_y;
			flow_data.squal = temp_data[4];
			 high = ((u32)temp_data[0]<<8) | ((u16)temp_data[1]) ;   //����߶�
			if(high!=0)
				VL53L01_high=high;
			error_count=0;
	}
	else
	{
			if(error_count<10)//���ͨ��ʧ�ܣ���Ҫ�˳����߶���
			{
				error_count++;
			}
			else
			{
					ALL_flag.height_lock=0;
					ALL_flag.flow_control=0;
			}
	}
	
}	


















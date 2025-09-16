#ifndef __FLOW_H_
#define __FLOW_H_
#include "stm32f10x.h"
//typedef struct st_optical_flow_data
//{
//s16 flow_x_integral;
////X ���ص��ۼ�ʱ���ڵ��ۼ�λ��,��Χ�� [-500 ~ =< value <= +500]
//s16 flow_y_integral;
////Y ���ص��ۼ�ʱ���ڵ��ۼ�λ�ƣ���Χ�� [-500 ~ =< value <= +500]
//u16 integration_timespan;
////���ζ��߹������ݵ���һ�ζ��߹������ݵ��ۼ�ʱ��(us).
//u16 ground_distance;
////������
//u8 quality; //�������ݵ�����,��ֵԽ������Խ��. [ 0 <value < 250 ]
//u8 version; //����ģ��Э��汾��
//}T_ST_Pixels_OpticalFLow_Data;


//extern T_ST_Pixels_OpticalFLow_Data flow_data;
//extern float flow_x_i;
//extern float flow_y_i;
extern float flow_x_lpf_att_i;
extern float flow_y_lpf_att_i;
extern float flow_x_vel_lpf_i;
extern float flow_y_vel_lpf_i;
extern void flow_data_sins(void);
extern void flow_sins_rest(void);
extern uint32_t VL53L01_high; //��ǰ�߶�
extern void flow_get_data(void);
typedef struct 
{
	int16_t vel_x;
	int16_t vel_y;
	float pos_x_i;
	float pos_y_i;	
	uint8_t squal;
	uint8_t end_flag;
}FLow;
extern FLow flow_data;

#endif


#include "usart.h"
#include "misc.h"
#include "stdio.h"
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#define BUFFER_SIZE 24  // ���������Ҫ������������С
/*
 * ��������USART1_Config
 * ����  ��USART1 GPIO ����,����ģʽ���á�
 * ����  ����
 * ���  : ��
 * ����  ���ⲿ����
 */
void USART1_Config(void)
{
	GPIO_InitTypeDef GPIO_InitStructure;
	USART_InitTypeDef USART_InitStructure;
	NVIC_InitTypeDef NVIC_InitStructure; 
	/* ���ô���1 ��USART1�� ʱ��*/
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1 | RCC_APB2Periph_GPIOA, ENABLE);
	
	/* Configure the NVIC Preemption Priority Bits */  

	
	/* ʹ�ܴ���1�ж� */
	NVIC_InitStructure.NVIC_IRQChannel = USART1_IRQn;	//USART1  ����1ȫ���ж� 
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;//��ռ���ȼ�1
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0; //�����ȼ�1
	/*IRQͨ��ʹ��*/
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
	/*����NVIC_InitStruct��ָ���Ĳ�����ʼ������NVIC�Ĵ���USART1*/
	NVIC_Init(&NVIC_InitStructure);
	
	

	/*����GPIO�˿�����*/
  /* ���ô���1 ��USART1 Tx (PA.09)��*/
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA, &GPIO_InitStructure);
  
	/* ���ô���1 USART1 Rx (PA.10)*/
  GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
	GPIO_Init(GPIOA, &GPIO_InitStructure);
	
	
	//USART ��ʼ������
	USART_InitStructure.USART_BaudRate = 115200;//���ڲ�����
	USART_InitStructure.USART_WordLength = USART_WordLength_8b;//�ֳ�Ϊ8λ���ݸ�ʽ
	USART_InitStructure.USART_StopBits = USART_StopBits_1;//һ��ֹͣλ
	USART_InitStructure.USART_Parity = USART_Parity_No;//����żУ��λ
	USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;//��Ӳ������������
	USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;	//�շ�ģʽ
	USART_Init(USART1, &USART_InitStructure);
	
	USART_ITConfig(USART1, USART_IT_RXNE, ENABLE);//�����ж�
	
	USART_Cmd(USART1, ENABLE); //ʹ�ܴ��� 
}
uint8_t receivedData[BUFFER_SIZE] = {0};
uint16_t value1 = 1500, value2 = 1500,Data_Point=0;
uint16_t value3 = 1500; uint16_t value4 = 1500; 
uint16_t rec_data=0,data_state=0;

uint16_t value5 = 0;

void parseData(uint8_t *data, uint16_t *val1, uint16_t *val2,uint16_t *val3,uint16_t *val4,uint16_t *val5) {
	*val1 = data[1]*1000 + data[2]*100 + data[3]*10 + data[4]*1;
	*val2 = data[6]*1000 + data[7]*100 + data[8]*10 + data[9]*1;
	*val3 = data[11]*1000 + data[12]*100 + data[13]*10 + data[14]*1;
	*val4 = data[16]*1000 + data[17]*100 + data[18]*10 + data[19]*1;
	*val5 = data[21];
	cnt_600ms=0;
}

// void parseData(uint8_t *data, uint16_t *val1, uint16_t *val2) {
// 	*val1 = data[1]*1000 + data[2]*100 + data[3]*10 + data[4]*1;
// 	*val2 = data[6]*1000 + data[7]*100 + data[8]*10 + data[9]*1;
// 	cnt_600ms=0;
// 	// printf("%d,%d#\r\n",*val1,*val2);
// }


void USART1_IRQHandler(void)//:1500,1500,1500,1500#
{
	if(USART_GetITStatus(USART1, USART_IT_RXNE))
	{
		
		rec_data=USART_ReceiveData(USART1);
		if(rec_data == ':' && data_state==0)
		{
			Data_Point=0;
			data_state=1;
		}
		if(data_state==1)
		{
			receivedData[Data_Point]=rec_data - '0';
			Data_Point++;
			if(rec_data == '#')
			{
				if(Data_Point == 23){
					parseData(receivedData, &value1, &value2, &value3, &value4, &value5);
				}
				memset(receivedData,'\0',BUFFER_SIZE);
				Data_Point=0;
				data_state=0;
			}
		}
		USART_ClearITPendingBit(USART1,USART_IT_RXNE); //����жϱ�־
	}
}


// void USART1_IRQHandler(void)
// {
// 	if(USART_GetITStatus(USART1, USART_IT_RXNE))
// 	{
		
// 		rec_data=USART_ReceiveData(USART1);
// 		USART_SendData(USART1,rec_data);
// 		while (USART_GetFlagStatus(USART1, USART_FLAG_TXE) == RESET);
// 		if(rec_data == ':' && data_state==0)
// 		{
// 			data_state=1;
// 		}
// 		if(data_state==1)
// 		{
// 			receivedData[Data_Point]=rec_data - '0';
// 			Data_Point++;
// 			if(rec_data == '#')
// 			{
// 				if(Data_Point == 11){
// 					parseData(receivedData, &value1, &value2);
					
// 					memset(receivedData,'\0',20);
// 				}
// 				else {
// 					// printf("�������ݴ���\r\n");
// 					memset(receivedData,'\0',20);
// 				}
// 				Data_Point=0;
// 				data_state=0;
// 			}
// 		}
// 		USART_ClearITPendingBit(USART1,USART_IT_RXNE); //����жϱ�־
// 	}
// }


void USART1_SendByte(const int8_t *Data,uint8_t len)
{ 
	uint8_t i;
	
	for(i=0;i<len;i++)
	{
		while (!(USART1->SR & USART_FLAG_TXE));	 // while(USART_GetFlagStatus(USART1, USART_FLAG_TXE) == RESET)
		USART_SendData(USART1,*(Data+i));	 
	}
}

//����ȥ������
//��pwmƵ�� 100ms
int8_t CheckSend[7]={0xAA,0XAA,0xEF,2,0,0,0};

void USART1_setBaudRate(uint32_t baudRate)
{
	USART_InitTypeDef USART_InitStructure;
	USART_InitStructure.USART_BaudRate =  baudRate;
	USART_Init(USART1, &USART_InitStructure);
}

/*
 * ��������fputc
 * ����  ���ض���c�⺯��printf��USART1
 * ����  ����
 * ���  ����
 * ����  ����printf����
 */
int fputc(int ch, FILE *f)
{
	/* ��Printf���ݷ������� */
	USART_SendData(USART1, (unsigned char) ch);
	while( USART_GetFlagStatus(USART1,USART_FLAG_TC)!= SET);	
	return (ch);
}


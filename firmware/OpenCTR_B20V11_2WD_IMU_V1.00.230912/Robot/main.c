/**			                                                    
		   ____                    _____ _______ _____       XTARK@���˴���
		  / __ \                  / ____|__   __|  __ \ 
		 | |  | |_ __   ___ _ __ | |       | |  | |__) |
		 | |  | | '_ \ / _ \ '_ \| |       | |  |  _  / 
		 | |__| | |_) |  __/ | | | |____   | |  | | \ \ 
		  \____/| .__/ \___|_| |_|\_____|  |_|  |_|  \_\
				| |                                     
				|_|                OpenCTR   �����˿�����
									 
  ****************************************************************************** 
  *           
  * ��Ȩ���У� XTARK@���˴���  ��Ȩ���У�����ؾ�
  * ��˾��վ�� www.xtark.cn   www.tarkbot.com
  * �Ա����̣� https://xtark.taobao.com  
  * ����΢�ţ� ���˴��£���ע���ںţ���ȡ���¸�����Ѷ��
  *           
  ******************************************************************************
  * @��  ��  Musk Han@XTARKXTARK
  * @��  ��  ��������
  * 
  ******************************************************************************
  */ 


/* Includes ------------------------------------------------------------------*/
#include "stm32f10x.h"
#include <stdio.h>
#include "ax_robot.h"

#include "ax_oled_chinese.h" //OLED���ֿ�
#include "ax_oled_picture.h" //OLED ͼƬ��

//������
//��������
#define START_TASK_PRIO		1
#define START_STK_SIZE 		128  
TaskHandle_t StartTask_Handler = NULL;
void Start_Task(void *pvParameters);

//����������������
#define ROBOT_TASK_PRIO		3     
#define ROBOT_STK_SIZE 		256 
TaskHandle_t Robot_Task_Handle = NULL;
void Robot_Task(void *pvParameters);

//������������
#define KEY_TASK_PRIO		4     
#define KEY_STK_SIZE 		128   
TaskHandle_t Key_Task_Handle = NULL;
void Key_Task(void *pvParameters);

//OLED��ʾ����
#define DISP_TASK_PRIO		6     
#define DISP_STK_SIZE 		128   
TaskHandle_t Disp_Task_Handle = NULL;
void Disp_Task(void *pvParameters);

//���¹�������
#define TRIVIA_TASK_PRIO		10     
#define TRIVIA_STK_SIZE 		128   
TaskHandle_t Trivia_Task_Handle = NULL;
void Trivia_Task(void *pvParameters);

//PS2���ݻ�ȡ����
#define PS2_TASK_PRIO		11     
#define PS2_STK_SIZE 		128   
TaskHandle_t Ps2_Task_Handle = NULL;
void Ps2_Task(void *pvParameters);


/**
  * @��  ��  ����������
  * @��  ��  ��
  * @����ֵ  ��
  */
int main(void)
{	

	//�����ж����ȼ�����
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);   

	//�����ʼ��
	AX_MOTOR_Init();
	
	//��ʱ������ʼ��
	AX_DELAY_Init();  
	
	//JTAG��ʼ��
	AX_JTAG_Set(JTAG_SWD_DISABLE);    	
	AX_JTAG_Set(SWD_ENABLE);      
    
	//LED��ʼ��
	AX_LED_Init();  
	
	//KEY��������ʼ��
	AX_KEY_Init();
	
	//��ص�ѹ����ʼ��
	AX_VIN_Init();
	
	//��������ʼ��
	AX_BEEP_Init();  
	
	//���Դ��ڳ�ʼ��
	AX_UART1_Init(230400);
	
	//�������ڳ�ʼ��
	AX_UART2_Init(115200);

	/* Raspberry Pi / host: X-Protocol on UART4 (PC10/PC11), 115200 8N1 */
	AX_UART4_Init(115200);
	
	//PS2�ֱ���ʼ��
	AX_PS2_Init();
	
	//��������ʼ��
	AX_ENCODER_A_Init();  
	AX_ENCODER_B_Init(); 
	
	//OLED��Ļ��ʼ��
	AX_OLED_Init();	
	AX_OLED_DispPicture(0, 0, 128, 8, PIC64X128_XTARK, 0); 
	
	
	//������ʾ��Ϣ
	AX_BEEP_On();
	AX_Delayms(100);	
	AX_BEEP_Off();
	AX_Delayms(1000);
	
	//��������������ʼ��
	AX_ULTRASONIC_Init();
	
	//MPU6050��ʼ��
	AX_MPU6050_Init();      
	AX_MPU6050_SetAccRange(AX_ACC_RANGE_2G);    //���ü��ٶ�����
	AX_MPU6050_SetGyroRange(AX_GYRO_RANGE_500); //��������������
	AX_MPU6050_SetGyroSmplRate(200);            //���������ǲ�����
	AX_MPU6050_SetDLPF(AX_DLPF_ACC94_GYRO98);   //���õ�ͨ�˲�������
	
	//����AppTaskCreate����
	xTaskCreate((TaskFunction_t )Start_Task,  /* ������ں��� */
								 (const char*    )"Start_Task",/* �������� */
								 (uint16_t       )START_STK_SIZE,  /* ����ջ��С */
								 (void*          )NULL,/* ������ں������� */
								 (UBaseType_t    )START_TASK_PRIO, /* ��������ȼ� */
								 (TaskHandle_t*  )&StartTask_Handler);/* ������ƿ�ָ�� */ 
							
	//�������񣬿�������						 
	vTaskStartScheduler(); 

	//ѭ��
	while (1);
}


/**
  * @��  ��  ����������
  * @��  ��  ��
  * @����ֵ  ��
  */
void Start_Task(void *pvParameters)
{
	//������У׼����
	int16_t gyro_data[3]; 
	
	//���������У׼
	for(int i=0; i<10; i++) 
	{
		//�����˸��ָʾ������У׼
		AX_LED_Green_On();
		vTaskDelay(30); 
		
		AX_LED_Green_Off();
		vTaskDelay(20); 

		//��ȡPMU6050����������
        AX_MPU6050_GetGyroData(gyro_data);
		
		ax_imu_gyro_offset[0] += gyro_data[0];
		ax_imu_gyro_offset[1] += gyro_data[1];
		ax_imu_gyro_offset[2] += gyro_data[2]; 		
	}
	
	//����ƽ��ƫ��ֵ
	ax_imu_gyro_offset[0] = -ax_imu_gyro_offset[0]/10;
	ax_imu_gyro_offset[1] = -ax_imu_gyro_offset[1]/10;
	ax_imu_gyro_offset[2] = -ax_imu_gyro_offset[2]/10;	
	
	/******������OLED��������ʾ************************************************/
	//��ʾ�����ڽ���
	AX_OLED_ClearScreen();  //���OLED����������ʾ
	AX_OLED_DispStr(0, 0, "   * TARKBOT TWD *   ", 0);	
	AX_OLED_DispStr(0, 1, "---------------------", 0);	
	AX_OLED_DispStr(0, 2, " MOD:FN1   Vol:12.2V ", 0);
	AX_OLED_DispStr(0, 3, " Gyz:00000           ", 0);
	AX_OLED_DispStr(0, 4, " USF:----  USR:----   ", 0);
	AX_OLED_DispStr(0, 5, "---------------------", 0);	
	AX_OLED_DispStr(0, 6, " MTA:00.00 MTB:-0.00 ", 0);
	
	//�����ٽ���
	taskENTER_CRITICAL();           
  
	//���������˿�������
	xTaskCreate((TaskFunction_t )Robot_Task, /* ������ں��� */
			 (const char*    )"Robot_Task",/* �������� */
			 (uint16_t       )ROBOT_STK_SIZE,   /* ����ջ��С */
			 (void*          )NULL,	/* ������ں������� */
			 (UBaseType_t    )ROBOT_TASK_PRIO,	    /* ��������ȼ� */
			 (TaskHandle_t*  )&Robot_Task_Handle);/* ������ƿ�ָ�� */
			 	 								 
	//����������������
	xTaskCreate((TaskFunction_t )Key_Task, /* ������ں��� */
			 (const char*    )"Key_Task",/* �������� */
			 (uint16_t       )KEY_STK_SIZE,   /* ����ջ��С */
			 (void*          )NULL,	/* ������ں������� */
			 (UBaseType_t    )KEY_TASK_PRIO,	    /* ��������ȼ� */
			 (TaskHandle_t*  )&Key_Task_Handle);/* ������ƿ�ָ�� */		

	//OLED����ʾ����
	xTaskCreate((TaskFunction_t )Disp_Task, /* ������ں��� */
			 (const char*    )"Disp_Task",/* �������� */
			 (uint16_t       )DISP_STK_SIZE,   /* ����ջ��С */
			 (void*          )NULL,	/* ������ں������� */
			 (UBaseType_t    )DISP_TASK_PRIO,	    /* ��������ȼ� */
			 (TaskHandle_t*  )&Disp_Task_Handle);/* ������ƿ�ָ�� */	
			 
	//���¹�������
	xTaskCreate((TaskFunction_t )Trivia_Task, /* ������ں��� */
			 (const char*    )"Trivia_Task",/* �������� */
			 (uint16_t       )TRIVIA_STK_SIZE,   /* ����ջ��С */
			 (void*          )NULL,	/* ������ں������� */
			 (UBaseType_t    )TRIVIA_TASK_PRIO,	    /* ��������ȼ� */
			 (TaskHandle_t*  )&Trivia_Task_Handle);/* ������ƿ�ָ�� */

	//PS2�ֱ����ݶ�ȡ����
	xTaskCreate((TaskFunction_t )Ps2_Task, /* ������ں��� */
			 (const char*    )"Ps2_Task",/* �������� */
			 (uint16_t       )PS2_STK_SIZE,   /* ����ջ��С */
			 (void*          )NULL,	/* ������ں������� */
			 (UBaseType_t    )PS2_TASK_PRIO,	    /* ��������ȼ� */
			 (TaskHandle_t*  )&Ps2_Task_Handle);/* ������ƿ�ָ�� */				 
			 
						  
	//ɾ��AppTaskCreate����				
	vTaskDelete(StartTask_Handler); 

	//�˳��ٽ���
	taskEXIT_CRITICAL();           
						 							
}

/******************* (C) ��Ȩ 2023 XTARK **************************************/


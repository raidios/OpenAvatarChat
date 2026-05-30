/**			                                                    
		   ____                    _____ _______ _____       @塔克创新
		  / __ \                  / ____|__   __|  __ \ 
		 | |  | |_ __   ___ _ __ | |       | |  | |__) |
		 | |  | | '_ \ / _ \ '_ \| |       | |  |  _  / 
		 | |__| | |_) |  __/ | | | |____   | |  | | \ \ 
		  \____/| .__/ \___|_| |_|\_____|  |_|  |_|  \_\
				| |                                     
				|_|                OpenCTR   机器人控制器
									 
  ****************************************************************************** 
  *           
  * 版权所有： @塔克创新  版权所有，盗版必究
  * 公司网站： www.xtark.cn   www.tarkbot.com
  * 淘宝店铺： https://xtark.taobao.com  
  * 塔克微信： 塔克创新（关注公众号，获取最新更新资讯）
  *      
  ******************************************************************************
  * @作  者  Musk Han@XTARK
  * @内  容  机器人运动学解析
  * 
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "ax_kinematics.h"
#include "ax_robot.h"
#include "ax_speed.h"


//根据机器人类型，选择编译机器人运动学处理函数


void ROBOT_TWD_Kinematics(void);

/**
  * @简  述  机器人运动学处理，不同类型底盘进入不同处理函数
  * @参  数  无
  * @返回值  无
  */
void AX_ROBOT_Kinematics(void)
{	
	
	//判断机器人运动是否开启
	if(ax_robot_move_enable == 0)
	{
		//目标速度设置为0
		R_Vel.TG_IX = 0;
		R_Vel.TG_IY = 0;
		R_Vel.TG_IW = 0;
	}
	
	/* Ultrasonic safety with debounce:
	   - Block immediately on ANY close reading (safety first)
	   - Only release after US_CLEAR_CYCLES consecutive safe readings */
	{
		static uint8_t us_front_clear_cnt = 0;
		static uint8_t us_rear_clear_cnt  = 0;
		#define US_CLEAR_CYCLES  5
		
		if(ax_ultrasonic_front_mm < US_SAFETY_DISTANCE_MM)
			us_front_clear_cnt = 0;
		else if(us_front_clear_cnt < US_CLEAR_CYCLES)
			us_front_clear_cnt++;
		
		if(us_front_clear_cnt < US_CLEAR_CYCLES && R_Vel.TG_IX > 0)
			R_Vel.TG_IX = 0;
		
		if(ax_ultrasonic_rear_mm < US_SAFETY_DISTANCE_MM)
			us_rear_clear_cnt = 0;
		else if(us_rear_clear_cnt < US_CLEAR_CYCLES)
			us_rear_clear_cnt++;
		
		if(us_rear_clear_cnt < US_CLEAR_CYCLES && R_Vel.TG_IX < 0)
			R_Vel.TG_IX = 0;
	}
	
	ROBOT_TWD_Kinematics();
}

//R5/R10系列运动学处理
#if (defined ROBOT_R5) || (defined ROBOT_R10)
/**
  * @简  述  机器人运动学处理-两轮差速
  * @参  数  无
  * @返回值  无
  */
void ROBOT_TWD_Kinematics(void)
{
	
	//通过编码器获取车轮实时转速m/s
	R_Wheel_A.RT = (float) ((int16_t)AX_ENCODER_A_GetCounter()*TWD_WHEEL_SCALE);
	AX_ENCODER_A_SetCounter(0);
	R_Wheel_B.RT = (float)-((int16_t)AX_ENCODER_B_GetCounter()*TWD_WHEEL_SCALE);
	AX_ENCODER_B_SetCounter(0);			
	
	//调试输出轮子转速
	//printf("@%f  %f   \r\n",R_Wheel_A.RT,R_Wheel_B.RT);
	
	//运动学正解析，由机器人轮子速度计算机器人速度
	R_Vel.RT_IX = ((R_Wheel_A.RT + R_Wheel_B.RT)/2)*1000;
	R_Vel.RT_IY = 0;
	R_Vel.RT_IW = ((-R_Wheel_A.RT + R_Wheel_B.RT)/TWD_WHEEL_BASE)*1000;		
	
	//机器人目标速度限制
	if( R_Vel.TG_IX > R_VX_LIMIT )    R_Vel.TG_IX = R_VX_LIMIT;
	if( R_Vel.TG_IX < (-R_VX_LIMIT))  R_Vel.TG_IX = (-R_VX_LIMIT);
	if( R_Vel.TG_IY > R_VY_LIMIT)     R_Vel.TG_IY = R_VY_LIMIT;
	if( R_Vel.TG_IY < (-R_VY_LIMIT))  R_Vel.TG_IY = (-R_VY_LIMIT);
	if( R_Vel.TG_IW > R_VW_LIMIT)     R_Vel.TG_IW = R_VW_LIMIT;
	if( R_Vel.TG_IW < (-R_VW_LIMIT))  R_Vel.TG_IW = (-R_VW_LIMIT);
	
	//目标速度转化为浮点类型
	R_Vel.TG_FX = R_Vel.TG_IX/1000.0;
	R_Vel.TG_FY = 0;
	R_Vel.TG_FW = R_Vel.TG_IW/1000.0;
	
	//运动学逆解析，由机器人目标速度计算电机轮子速度（m/s）
	R_Wheel_A.TG = R_Vel.TG_FX - R_Vel.TG_FW*(TWD_WHEEL_BASE/2);
	R_Wheel_B.TG = R_Vel.TG_FX + R_Vel.TG_FW*(TWD_WHEEL_BASE/2);	
	

	//利用PID算法计算电机PWM值
	R_Wheel_A.PWM = AX_SPEED_PidCtlA(R_Wheel_A.TG, R_Wheel_A.RT);   
	R_Wheel_B.PWM = AX_SPEED_PidCtlB(R_Wheel_B.TG, R_Wheel_B.RT);  
 
	//设置电机PWM值
	AX_MOTOR_A_SetSpeed( -R_Wheel_A.PWM);
	AX_MOTOR_B_SetSpeed( -R_Wheel_B.PWM); 	
 
	
	//printf("A%f B%f  \r\n ",MOTOR_A.Wheel_RT, MOTOR_B.Wheel_RT  );
	//printf("A%d B%d C%d  \r\n ",R_Vel.I_X, R_Vel.I_Y, R_Vel.I_W );	
	
}
#endif


//TT系列运动学处理
#ifdef ROBOT_TT
/**
  * @简  述  机器人运动学处理-两轮差速
  * @参  数  无
  * @返回值  无
  */
void ROBOT_TWD_Kinematics(void)
{
	
	//通过编码器获取车轮实时转速m/s
	R_Wheel_A.RT = (float)-((int16_t)AX_ENCODER_A_GetCounter()*TWD_WHEEL_SCALE);
	AX_ENCODER_A_SetCounter(0);
	R_Wheel_B.RT = (float) ((int16_t)AX_ENCODER_B_GetCounter()*TWD_WHEEL_SCALE);
	AX_ENCODER_B_SetCounter(0);			
	
	//调试输出轮子转速
	//printf("@%f  %f   \r\n",R_Wheel_A.RT,R_Wheel_B.RT);
	
	//运动学正解析，由机器人轮子速度计算机器人速度
	R_Vel.RT_IX = ((R_Wheel_A.RT + R_Wheel_B.RT)/2)*1000;
	R_Vel.RT_IY = 0;
	R_Vel.RT_IW = ((-R_Wheel_A.RT + R_Wheel_B.RT)/TWD_WHEEL_BASE)*1000;		
	
	//机器人目标速度限制
	if( R_Vel.TG_IX > R_VX_LIMIT )    R_Vel.TG_IX = R_VX_LIMIT;
	if( R_Vel.TG_IX < (-R_VX_LIMIT))  R_Vel.TG_IX = (-R_VX_LIMIT);
	if( R_Vel.TG_IY > R_VY_LIMIT)     R_Vel.TG_IY = R_VY_LIMIT;
	if( R_Vel.TG_IY < (-R_VY_LIMIT))  R_Vel.TG_IY = (-R_VY_LIMIT);
	if( R_Vel.TG_IW > R_VW_LIMIT)     R_Vel.TG_IW = R_VW_LIMIT;
	if( R_Vel.TG_IW < (-R_VW_LIMIT))  R_Vel.TG_IW = (-R_VW_LIMIT);
	
	//目标速度转化为浮点类型
	R_Vel.TG_FX = R_Vel.TG_IX/1000.0;
	R_Vel.TG_FY = 0;
	R_Vel.TG_FW = R_Vel.TG_IW/1000.0;
	
	//运动学逆解析，由机器人目标速度计算电机轮子速度（m/s）
	R_Wheel_A.TG = R_Vel.TG_FX - R_Vel.TG_FW*(TWD_WHEEL_BASE/2);
	R_Wheel_B.TG = R_Vel.TG_FX + R_Vel.TG_FW*(TWD_WHEEL_BASE/2);	
	

	//利用PID算法计算电机PWM值
	R_Wheel_A.PWM = AX_SPEED_PidCtlA(R_Wheel_A.TG, R_Wheel_A.RT);   
	R_Wheel_B.PWM = AX_SPEED_PidCtlB(R_Wheel_B.TG, R_Wheel_B.RT);  
 
	//设置电机PWM值
	AX_MOTOR_A_SetSpeed( -R_Wheel_A.PWM);
	AX_MOTOR_B_SetSpeed(  R_Wheel_B.PWM); 	
 
	
	//printf("A%f B%f  \r\n ",MOTOR_A.Wheel_RT, MOTOR_B.Wheel_RT  );
	//printf("A%d B%d C%d  \r\n ",R_Vel.I_X, R_Vel.I_Y, R_Vel.I_W );	
	
}
#endif



/******************* (C) 版权 2023 XTARK **************************************/


/**			                                                    
		   ____                    _____ _______ _____       XTARK@塔克创新
		  / __ \                  / ____|__   __|  __ \ 
		 | |  | |_ __   ___ _ __ | |       | |  | |__) |
		 | |  | | '_ \ / _ \ '_ \| |       | |  |  _  / 
		 | |__| | |_) |  __/ | | | |____   | |  | | \ \ 
		  \____/| .__/ \___|_| |_|\_____|  |_|  |_|  \_\
				| |                                     
				|_|                OpenCTR   机器人控制器
									 
  ****************************************************************************** 
  *           
  * 版权所有： XTARK@塔克创新  版权所有，盗版必究
  * 公司网站： www.xtark.cn   www.tarkbot.com
  * 淘宝店铺： https://xtark.taobao.com  
  * 塔克微信： 塔克创新（关注公众号，获取最新更新资讯）
  *      
  ******************************************************************************
  * @作  者  Musk Han@XTARK
  * @内  容  机器人控制处理文件
  * 
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "ax_control.h"
#include "ax_robot.h"

/**
  * @简  述  处理PS2手柄控制命令
  * @参  数  无
  * @返回值  无
  */
void AX_CTL_Ps2(void)
{
	static uint8_t btn_joyl_flag = 0;
	static uint8_t btn_joyr_flag = 0;
	
	static uint8_t  speed = 4;
	
	//红绿灯模式下，执行控制操作
	if(my_joystick.mode ==  0x73)
	{
		R_Vel.TG_IX = (int16_t)(speed*(0x80 - my_joystick.RJoy_UD));
		R_Vel.TG_IY = (int16_t)(speed*(0x80 - my_joystick.RJoy_LR));
		R_Vel.TG_IW = (int16_t)(4*speed*(0x80 - my_joystick.LJoy_LR));
		
		//左摇杆按键，减速
		if(my_joystick.btn1 & PS2_BT1_JOY_L)
		{
			btn_joyl_flag = 1;
		}
		else
		{
			if(btn_joyl_flag)
			{
				
				//速度减小
				if(speed > 2)
				{
					speed--;
				}
				else
				{
					speed = 2;
					
					//蜂鸣器鸣叫提示
					ax_beep_ring = BEEP_SHORT;
				}
					
				//复位标记
				btn_joyl_flag = 0;
			}
		}
		
		//右摇杆按键，加速
		if(my_joystick.btn1 & PS2_BT1_JOY_R)
		{
			btn_joyr_flag = 1;
		}
		else
		{
			if(btn_joyr_flag)
			{
				//速度增加
				if(speed < 9)
				{
					speed++;
				}
				else
				{
					speed = 9;
					
					//蜂鸣器鸣叫提示
					ax_beep_ring = BEEP_SHORT;					
				}
					
				//复位标记
				btn_joyr_flag = 0;
			}
		}
	}
}	

/**
  * @简  述  处理手机APP控制命令
  * @参  数  无
  * @返回值  无
  */
void AX_CTL_App(void)
{
	static uint8_t comdata[16];
	
	//接收蓝牙APP串口数据
	if(AX_UART2_GetData(comdata))
	{
		//摇杆模式运动控制帧
		if((comdata[0] == ID_BLERX_YG))
		{
			R_Vel.TG_IX = (int16_t)(  6*(int8_t)comdata[4] );
			R_Vel.TG_IY = (int16_t)( -6*(int8_t)comdata[3] );
			R_Vel.TG_IW = (int16_t)(-20*(int8_t)comdata[1] );
		}
		
		//手柄模式运动控制帧
		if((comdata[0] == ID_BLERX_SB))
		{

			R_Vel.TG_IX = (int16_t)(  6*(int8_t)comdata[4] );
			R_Vel.TG_IY = (int16_t)( -6*(int8_t)comdata[3] );
			R_Vel.TG_IW = (int16_t)(-20*(int8_t)comdata[1] );
		}
	}
}

/******************* (C) 版权 2023 XTARK **************************************/


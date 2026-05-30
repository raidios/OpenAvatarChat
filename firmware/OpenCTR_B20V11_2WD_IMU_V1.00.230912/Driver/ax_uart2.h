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
  * @内  容  蓝牙串口通信
  *
  ******************************************************************************
  */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __AX_UART_PI_H
#define __AX_UART_PI_H

/* Includes ------------------------------------------------------------------*/	 
#include "stm32f10x.h"

//手柄模式按键定义
#define  APP_JOY1_Y         0x01    //Y按键
#define  APP_JOY1_X         0x02    //X按键
#define  APP_JOY1_B         0x04    //B按键 
#define  APP_JOY1_A         0x08    //A按键
#define  APP_JOY1_RIGHT     0x10    //方向 右按键
#define  APP_JOY1_LEFT      0x20    //方向 左按键
#define  APP_JOY1_DOWN      0x40    //方向 下按键 
#define  APP_JOY1_UP        0x80    //方向 上按键

#define  APP_JOY2_K4         0x01    //K1按键
#define  APP_JOY2_K3         0x02    //K2按键
#define  APP_JOY2_K2         0x04    //K3按键 
#define  APP_JOY2_K1         0x08    //K4按键
#define  APP_JOY2_R2         0x10    //R2按键
#define  APP_JOY2_R1         0x20    //R1按键
#define  APP_JOY2_L2         0x40    //L2按键 
#define  APP_JOY2_L1         0x80    //L1按键

//接口函数
void AX_UART2_Init(uint32_t baud);  //扩展串口初始化
uint8_t AX_UART2_GetData(uint8_t *pbuf);
void AX_UART2_SendPacket(uint8_t *pbuf, uint8_t len, uint8_t num);  //发送数据（X-Protocol协议）

#endif 

/******************* (C) 版权 2023 XTARK **************************************/

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
  * @内  容  4路红外巡线传感器
  ******************************************************************************
  * @说  明
  *
  * 
  ******************************************************************************
  */

#include "ax_line.h"
#include "ax_sys.h"
#include <stdio.h>


/**
  * @简  述  4路巡线传感器初始化
  * @参  数  无
  * @返回值  无
  */
void AX_LINE_Init(void)
{
	GPIO_InitTypeDef GPIO_InitStructure;
	
	// GPIO配置
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB,ENABLE);//启动GPIO时钟 
	
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_12 | GPIO_Pin_13 | GPIO_Pin_14 | GPIO_Pin_15;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU; 
 	GPIO_Init(GPIOB, &GPIO_InitStructure);
}


/**
  * @简  述  获取一次检测数据
  * @参  数  无
  *			L1(B15) L2(B14) L3(B13) L4(B12) 返回值
  *			 0        0       0       0      0 
  *          1        0       0       0      1
  *          0        1       0       0      2
  *                         .
  *          1        1       1       1      15   
  */
uint8_t AX_LINE_GetData(void)
{
	
    uint8_t re = 0;

	if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_12) == 0)
		re=re+1;
	re<<=1;
	
	if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_13) == 0)
		re=re+1;
	re<<=1;
	
	if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_14) == 0)
		re=re+1;
	re<<=1;
	
	if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_15) == 0)
		re=re+1;
	
	return re;	
}


/******************* (C) 版权 2023 XTARK **************************************/

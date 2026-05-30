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
  * @内  容  CCD巡线传感器
  ******************************************************************************
  * @说  明
  *
  * 
  ******************************************************************************
  */

#include "ax_ccd.h"
#include "ax_sys.h"
#include <stdio.h>

//引脚定义
#define TSL_CLK   PBout(11)   //CLK 
#define TSL_SI    PBout(10)   //SI  

//内部函数定义
static void CCD_Delay_us(void);  //CCD专用延时函数
static uint16_t CCD_GetAdcData(void);  //获取ADC采样值

/**
  * @简  述  CCD巡线传感器初始化
  * @参  数  无
  * @返回值  无
  */
void AX_CCD_Init(void)
{
	GPIO_InitTypeDef GPIO_InitStructure;
	ADC_InitTypeDef ADC_InitStructure;
	
	//配置GPIO AO
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);

	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_1;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;
	GPIO_Init(GPIOA, &GPIO_InitStructure);
	
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);

	//配置GPIO SI,CLK
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10 | GPIO_Pin_11;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;	
	GPIO_Init(GPIOB, &GPIO_InitStructure);	
	
	//**配置ADC******
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_ADC1, ENABLE);

	//ADC模式配置
	ADC_InitStructure.ADC_Mode = ADC_Mode_Independent;
	ADC_InitStructure.ADC_ScanConvMode = DISABLE;
	ADC_InitStructure.ADC_ContinuousConvMode = DISABLE;
	ADC_InitStructure.ADC_ExternalTrigConv = ADC_ExternalTrigConv_None;
	ADC_InitStructure.ADC_DataAlign = ADC_DataAlign_Right;
	ADC_InitStructure.ADC_NbrOfChannel = 1;
	ADC_Init(ADC1, &ADC_InitStructure);	

	//设置ADC分频因子6 72M/6=12,ADC最大时间不能超过14M
	RCC_ADCCLKConfig(RCC_PCLK2_Div6);

	//配置ADC通道、转换顺序和采样时间
	ADC_RegularChannelConfig(ADC1, ADC_Channel_1, 1, ADC_SampleTime_239Cycles5);

	//使能ADC
	ADC_Cmd(ADC1, ENABLE);

	//初始化ADC 校准寄存器  
	ADC_ResetCalibration(ADC1);

	//等待校准寄存器初始化完成
	while(ADC_GetResetCalibrationStatus(ADC1));

	//ADC开始校准
	ADC_StartCalibration(ADC1);

	//等待校准完成
	while(ADC_GetCalibrationStatus(ADC1)); 		
}

/**
  * @简  述  CCD获取一帧数据
  * @参  数  *pbuf：CCD一帧数据，数据个数128
  * @返回值  数据结果。
  */
void AX_CCD_GetData(uint16_t *pbuf)
{
	uint8_t i;
	
	//设置初始状态
	TSL_CLK=1;
    TSL_SI=0; 
    CCD_Delay_us();
    
	//启动采样和输出
    TSL_SI=1; 
    TSL_CLK=0;
    CCD_Delay_us();
	
	//设置等待采样
    TSL_CLK=1;
    TSL_SI=0;
    CCD_Delay_us(); 
	
	//读取128个像素点电压值
	for(i=0;i<128;i++)					
	{   
		//下降沿启动采样
		TSL_CLK=0; 
		CCD_Delay_us();  //调节曝光时间
        
		//获取转换结果
		*(pbuf+i) = (CCD_GetAdcData())>>4;
		TSL_CLK=1;
		CCD_Delay_us();	
	}
}

/**
  * @简  述  获取并计算黑线距离中间位置偏差
  * @参  数  无
  * @返回值  偏差结果，中间为0，范围±64。
  */
int16_t AX_CCD_GetOffset(void)
{
	uint8_t i;
	static uint16_t ccd[128]= {0};
	static uint16_t ccd_min,ccd_max;
	static uint16_t ccd_threshold;
	static uint16_t ccd_left,ccd_right;
	static int16_t ccd_offset;
	
	//获取CCD一帧的128个数据
	AX_CCD_GetData(ccd);
	
//	//调试使用，查看CCD原始数据	
//	for(i=0; i<128; i++) printf("%d ",ccd[i]);
//	printf("\r\n");

	//CCD采集128个像素数据，每个数据与阈值进行比较，比阈值大为白色，比阈值小为黑色
	//动态阈值算法，一帧数据的最大和最小值的中间值作为阈值 
	
	//-----计算动态阈值-------
	ccd_min = ccd[0];
	ccd_max = ccd[0];
	
	//遍历数组，寻找最值
	for(i=0; i<128; i++)
	{
		//寻找最小值
		if(ccd_min > ccd[i])
		{
			ccd_min = ccd[i];
		}
		
		//寻找最大值
		if(ccd_max < ccd[i])
		{
			ccd_max = ccd[i];
		}	
	}
	
	//计算阈值
	ccd_threshold =(ccd_min + (ccd_max-ccd_min)/2);	  
	
	
	//查找黑线左边变沿，白像素后连续4个黑像素判断变沿
	for(i=0; i<125; i++)
	{
		//
		if(ccd[i] < ccd_threshold)
		{
			//后方三个像素连续为黑像素
			if((ccd[i+1]<ccd_threshold) && (ccd[i+2]<ccd_threshold) && (ccd[i+3]<ccd_threshold))
			{
				ccd_left = i;
				break;
			}
		}
	}
	
	//查找黑线右边变沿，白像素前方连续4个黑像素判断变沿
	for(i=128; i>2; i--)
	{
		//
		if(ccd[i] < ccd_threshold)
		{
			//前方三个像素连续为黑线
			if((ccd[i-1]<ccd_threshold) && (ccd[i-2]<ccd_threshold) && (ccd[i-3]<ccd_threshold))
			{
				ccd_right = i;
				break;
			}
		}
	}
	
	//计算偏差值
	ccd_offset = (ccd_left+ccd_right)/2 - 64;
	
	//打印调试信息
	//printf("%d  %d \r\n",ccd_threshold, ccd_offset);
	
	return ccd_offset;
}

/**
  * @简  述  CCD获取ADC测量值
  * @参  数  无	  
  * @返回值  ADC转换结果。
  */
static uint16_t CCD_GetAdcData(void)
{
	//软件启动转换功能 
	ADC_SoftwareStartConvCmd(ADC1, ENABLE);  

	//等待转换结束 
	while(!ADC_GetFlagStatus(ADC1, ADC_FLAG_EOC ));
	
	return ADC_GetConversionValue(ADC1);
}

/**
  * @简  述  CCD专用延时函数
  * @参  数  无	  
  * @返回值  无
  */
static void CCD_Delay_us(void)
{
   int i;   

   for(i=0;i<30;i++);      
}



/******************* (C) 版权 2023 XTARK **************************************/

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
  * @内  容  机器人功能处理文件
  * 
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "ax_function.h"
#include "ax_robot.h"

/**
  * @简  述  功能1-陀螺仪闭环控制直线行驶
  * @参  数  无
  * @返回值  无
  */
void AX_FUN_Ls1(void)
{
	int16_t gyro_z;
	static float bias;
	float move_w=0;   
	
	//直行速度，0.2m/s
	R_Vel.TG_IX =  200;
	
	//获取PMU6050陀螺仪数据
	AX_MPU6050_GetGyroData(ax_imu_gyro_data);
	
	//陀螺仪加入零票校准数据
	ax_imu_gyro_data[0] += ax_imu_gyro_offset[0];
	ax_imu_gyro_data[1] += ax_imu_gyro_offset[1];
	ax_imu_gyro_data[2] += ax_imu_gyro_offset[2];
	
	gyro_z = ax_imu_gyro_data[2];
	
	//陀螺仪积分，计算角度偏差
	bias +=  gyro_z;
	
	//PID计算输出
	move_w = -ax_imu_kp*bias*0.001f - ax_imu_kd*(gyro_z)*0.001f;
	
	//赋值给转向速度
	R_Vel.TG_IW = move_w;	
	
	
	//打印调试信息
	//printf("@%d %d %f \r\n", gyro_z, bias, move_w);
}





/******************* (C) 版权 2023 XTARK **************************************/


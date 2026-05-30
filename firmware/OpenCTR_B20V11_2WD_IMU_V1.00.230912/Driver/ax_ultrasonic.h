/**			                                                    
		   ____                    _____ _______ _____       
		  / __ \                  / ____|__   __|  __ \ 
		 | |  | |_ __   ___ _ __ | |       | |  | |__) |
		 | |  | | '_ \ / _ \ '_ \| |       | |  |  _  / 
		 | |__| | |_) |  __/ | | | |____   | |  | | \ \ 
		  \____/| .__/ \___|_| |_|\_____|  |_|  |_|  \_\
				| |                                     
				|_|                OpenCTR
									 
  ****************************************************************************** 
  * Ultrasonic sensor driver (HC-SR04)
  * Front sensor: Trig=PB12, Echo=PB13  (line-sensor connector)
  * Rear  sensor: Trig=PB14, Echo=PB15  (line-sensor connector)
  ******************************************************************************
  */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __AX_ULTRASONIC_H
#define __AX_ULTRASONIC_H

/* Includes ------------------------------------------------------------------*/	 
#include "stm32f10x.h"

/* Safety distance threshold (mm). Robot stops when obstacle closer than this */
#define US_SAFETY_DISTANCE_MM  250

/* Measurement timeout (us). ~1m max range */
#define US_TIMEOUT_US  6000

/* Distance readings (mm). 0xFFFF = timeout / no obstacle detected */
extern uint16_t ax_ultrasonic_front_mm;
extern uint16_t ax_ultrasonic_rear_mm;

void AX_ULTRASONIC_Init(void);
void AX_ULTRASONIC_Update(void);

#endif 

/***************************************************************************/

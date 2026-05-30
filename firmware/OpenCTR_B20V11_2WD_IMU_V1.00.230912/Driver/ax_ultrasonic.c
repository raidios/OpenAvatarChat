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
  *
  * Pin assignment (on line-sensor connectors):
  *   Front sensor: Trig = PB12, Echo = PB13
  *   Rear  sensor: Trig = PB14, Echo = PB15
  *
  * VCC/GND must be wired separately (e.g. from a servo or UART connector).
  *
  * Measurement uses SysTick hardware counter for microsecond-level timing.
  * AX_ULTRASONIC_Update() alternates between front and rear sensors each
  * call, so each sensor is refreshed at half the calling frequency.
  ******************************************************************************
  */

#include "ax_ultrasonic.h"
#include "ax_delay.h"

/* ---------- Pin definitions ------------------------------------------------*/
#define US_FRONT_TRIG_PORT  GPIOB
#define US_FRONT_TRIG_PIN   GPIO_Pin_12
#define US_FRONT_ECHO_PORT  GPIOB
#define US_FRONT_ECHO_PIN   GPIO_Pin_13

#define US_REAR_TRIG_PORT   GPIOB
#define US_REAR_TRIG_PIN    GPIO_Pin_14
#define US_REAR_ECHO_PORT   GPIOB
#define US_REAR_ECHO_PIN    GPIO_Pin_15

/* ---------- Global variables -----------------------------------------------*/
uint16_t ax_ultrasonic_front_mm = 0xFFFF;
uint16_t ax_ultrasonic_rear_mm  = 0xFFFF;

/* ---------- Private variables ----------------------------------------------*/
static uint8_t us_cycle = 0;

/* ---------- Private functions ----------------------------------------------*/

/**
  * @brief  Measure distance for one HC-SR04 sensor (blocking, with timeout)
  * @param  trig_port / trig_pin : Trigger GPIO
  * @param  echo_port / echo_pin : Echo GPIO
  * @retval Distance in mm, or 0xFFFF on timeout
  */
static uint16_t AX_ULTRASONIC_MeasureOne(
	GPIO_TypeDef* trig_port, uint16_t trig_pin,
	GPIO_TypeDef* echo_port, uint16_t echo_pin)
{
	uint32_t told, tnow, tcnt;
	uint32_t reload = SysTick->LOAD;
	uint32_t fac = SystemCoreClock / 1000000;          /* ticks per us */
	uint32_t timeout_ticks = (uint32_t)US_TIMEOUT_US * fac;
	uint32_t us;

	/* 1. Send >= 10 us trigger pulse */
	GPIO_SetBits(trig_port, trig_pin);
	AX_Delayus(12);
	GPIO_ResetBits(trig_port, trig_pin);

	/* 2. Wait for Echo rising edge (timeout protected) */
	told = SysTick->VAL;
	tcnt = 0;
	while (GPIO_ReadInputDataBit(echo_port, echo_pin) == Bit_RESET)
	{
		tnow = SysTick->VAL;
		if (tnow != told)
		{
			if (tnow < told)
				tcnt += told - tnow;
			else
				tcnt += reload - tnow + told;
			told = tnow;
			if (tcnt >= timeout_ticks)
				return 0xFFFF;
		}
	}

	/* 3. Measure Echo HIGH duration (= round-trip time of sound) */
	told = SysTick->VAL;
	tcnt = 0;
	while (GPIO_ReadInputDataBit(echo_port, echo_pin) == Bit_SET)
	{
		tnow = SysTick->VAL;
		if (tnow != told)
		{
			if (tnow < told)
				tcnt += told - tnow;
			else
				tcnt += reload - tnow + told;
			told = tnow;
			if (tcnt >= timeout_ticks)
				return 0xFFFF;
		}
	}

	/* 4. Convert to millimetres: distance_mm = us * 0.17  (speed of sound 340m/s) */
	us = tcnt / fac;
	return (uint16_t)(us * 17 / 100);
}

/* ---------- Public functions -----------------------------------------------*/

/**
  * @brief  Initialise ultrasonic sensor GPIOs
  */
void AX_ULTRASONIC_Init(void)
{
	GPIO_InitTypeDef GPIO_InitStructure;

	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);

	/* Trig pins: push-pull output */
	GPIO_InitStructure.GPIO_Mode  = GPIO_Mode_Out_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_InitStructure.GPIO_Pin   = US_FRONT_TRIG_PIN | US_REAR_TRIG_PIN;
	GPIO_Init(GPIOB, &GPIO_InitStructure);

	/* Echo pins: floating input (HC-SR04 echo is active-driven) */
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
	GPIO_InitStructure.GPIO_Pin  = US_FRONT_ECHO_PIN | US_REAR_ECHO_PIN;
	GPIO_Init(GPIOB, &GPIO_InitStructure);

	/* Trig idle LOW */
	GPIO_ResetBits(GPIOB, US_FRONT_TRIG_PIN | US_REAR_TRIG_PIN);
}

/**
  * @brief  Trigger and read one sensor per call (alternating front / rear).
  *         Call from Robot_Task at 50 Hz => each sensor updates at 25 Hz.
  *         Worst-case blocking time: US_TIMEOUT_US (~6 ms).
  */
void AX_ULTRASONIC_Update(void)
{
	if (us_cycle == 0)
	{
		ax_ultrasonic_front_mm = AX_ULTRASONIC_MeasureOne(
			US_FRONT_TRIG_PORT, US_FRONT_TRIG_PIN,
			US_FRONT_ECHO_PORT, US_FRONT_ECHO_PIN);
		us_cycle = 1;
	}
	else
	{
		ax_ultrasonic_rear_mm = AX_ULTRASONIC_MeasureOne(
			US_REAR_TRIG_PORT, US_REAR_TRIG_PIN,
			US_REAR_ECHO_PORT, US_REAR_ECHO_PIN);
		us_cycle = 0;
	}
}

/***************************************************************************/

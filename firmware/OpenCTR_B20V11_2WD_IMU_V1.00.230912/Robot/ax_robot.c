/**			                                                    
		   ____                    _____ _______ _____       @???????
		  / __ \                  / ____|__   __|  __ \ 
		 | |  | |_ __   ___ _ __ | |       | |  | |__) |
		 | |  | | '_ \ / _ \ '_ \| |       | |  |  _  / 
		 | |__| | |_) |  __/ | | | |____   | |  | | \ \ 
		  \____/| .__/ \___|_| |_|\_____|  |_|  |_|  \_\
				| |                                     
				|_|                OpenCTR   ???????????
									 
  ****************************************************************************** 
  *           
  * ??????��? @???????  ??????��???????
  * ???????? www.xtark.cn   www.tarkbot.com
  * ???????? https://xtark.taobao.com  
  * ???????? ????????????????????????????????
  *           
  ******************************************************************************
  * @??  ??  Musk Han@XTARK
  * @??  ??  ???????????????
  * 
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "ax_robot.h"
#include "ax_light.h"
#include "ax_kinematics.h"
#include "ax_control.h"
#include "ax_function.h"

/* ---- Input-arbitration tunables -------------------------------------------
 * PS2 is the highest-priority "fallback" controller: any user activity on the
 * pad instantly overrides the Pi/ROS cmd_vel stream. PS2 control is only
 * yielded back to the Pi once the sticks have been re-centred AND no buttons
 * are pressed for PS2_RELEASE_MS.
 *
 *   PS2_DEADZONE        - analog-stick distance from 0x80 (center) below which
 *                         we consider the stick "not touched". 0x80 == 128, so
 *                         a deadzone of 30 ignores the bottom ~23% on each side
 *                         and tolerates moderate drift / cheap-clone noise.
 *   PS2_RELEASE_MS      - how long PS2 must look idle before Pi can take over.
 *   PS2_WARMUP_MS       - PS2 is forbidden from grabbing the wheels for this
 *                         long after Robot_Task starts. Knock-off receivers
 *                         frequently spit out garbage frames in their first
 *                         few hundred milliseconds and would otherwise pin the
 *                         throttle on boot.
 */
#define PS2_DEADZONE     30
#define PS2_RELEASE_MS   800u
#define PS2_WARMUP_MS    2000u

/* Pi cmd_vel is considered "live" for this many ms after the most recent
 * 0x50 frame. Beyond it we stop the wheels (or fall through to APP/FN1).
 */
#define PI_CMD_TIMEOUT_MS 300u

/* Currently active input source, mirrored to the host via UART4 CMD 0x12. */
uint8_t ax_ctrl_source = AX_CTRL_SRC_IDLE;

//?????????????
ROBOT_Velocity  R_Vel;

//??????????????
ROBOT_Wheel  R_Wheel_A,R_Wheel_B;

//?????????????
ROBOT_Velocity  R_Vel;

//??????????????
uint16_t R_Bat_Vol;  

//???PID???????
int16_t ax_motor_kp=800;      
int16_t ax_motor_kd=1000; 

//???????????????
uint8_t ax_robot_move_enable = 0;

//?????????????
uint8_t ax_beep_ring = 0;

//PS2??????????
JOYSTICK_TypeDef my_joystick;  

/* Default control mode is "no auto-mode": neither FN1 (gyro-locked autostraight)
 * nor APP (Bluetooth) gets to drive the wheels until the user explicitly opts
 * in. The previous default of CTL_FN1 caused the robot to auto-roll forward at
 * 0.2 m/s the moment the Pi cmd_vel stream paused for >300 ms (see
 * AX_FUN_Ls1() in ax_function.c). */
uint8_t ax_control_mode = 0;

//IMU????
int16_t ax_imu_acc_data[3];  
int16_t ax_imu_gyro_data[3]; 
int16_t ax_imu_gyro_offset[3]; 

//??????PID????
int16_t ax_imu_kp=50, ax_imu_kd=50;

/**
 * @brief  Detects "all four sticks pinned to 0x00 or 0xFF" - typical of a
 *         knock-off PS2 receiver that sets mode=0x73 before it has actually
 *         negotiated with a controller, or of an unconnected DI line latched
 *         high by the IPU pull-up. Treat it as "no input" so the wheels stay
 *         still on boot.
 */
static uint8_t PS2_LooksLikeGlitch(void)
{
	uint8_t a = my_joystick.RJoy_UD;
	uint8_t b = my_joystick.RJoy_LR;
	uint8_t c = my_joystick.LJoy_UD;
	uint8_t d = my_joystick.LJoy_LR;

	if (a == 0xFF && b == 0xFF && c == 0xFF && d == 0xFF) return 1;
	if (a == 0x00 && b == 0x00 && c == 0x00 && d == 0x00) return 1;
	return 0;
}

/**
 * @brief  Returns 1 when the PS2 pad is being touched by a human.
 *         Triggers on any non-default button bit OR any analog-stick deflection
 *         beyond PS2_DEADZONE. Only the analog (red-LED, 0x73) mode counts as
 *         active control input; in digital mode the analog axes are pinned to
 *         0x80 anyway, so buttons are still detected via btn1/btn2.
 *
 *         Returns 0 unconditionally if the frame looks like a receiver
 *         glitch (PS2_LooksLikeGlitch).
 */
static uint8_t PS2_IsActive(void)
{
	int16_t lx, ly, rx, ry;

	if (PS2_LooksLikeGlitch()) return 0;

	/* btn1/btn2 are inverted in AX_PS2_ScanKey: 0x00 == nothing pressed. */
	if (my_joystick.btn1 != 0x00) return 1;
	if (my_joystick.btn2 != 0x00) return 1;

	if (my_joystick.mode == 0x73) {
		lx = (int16_t)my_joystick.LJoy_LR - 0x80;
		ly = (int16_t)my_joystick.LJoy_UD - 0x80;
		rx = (int16_t)my_joystick.RJoy_LR - 0x80;
		ry = (int16_t)my_joystick.RJoy_UD - 0x80;

		if (lx >  PS2_DEADZONE || lx < -PS2_DEADZONE) return 1;
		if (ly >  PS2_DEADZONE || ly < -PS2_DEADZONE) return 1;
		if (rx >  PS2_DEADZONE || rx < -PS2_DEADZONE) return 1;
		if (ry >  PS2_DEADZONE || ry < -PS2_DEADZONE) return 1;
	}

	return 0;
}

/**
 * @brief  Deadzone-aware reimplementation of AX_CTL_Ps2() used inside the
 *         arbitration override branch. Unlike the legacy ax_control.c version
 *         this one snaps any stick deflection within PS2_DEADZONE to zero, so
 *         a slightly mis-centered analog stick can no longer drive the wheels
 *         on its own. L-stick / R-stick clicks still trim the speed gain.
 */
static void Robot_ApplyPs2(void)
{
	static uint8_t btn_joyl_flag = 0;
	static uint8_t btn_joyr_flag = 0;
	static uint8_t speed         = 4;

	int16_t ud, lr, ll;

	if (my_joystick.mode == 0x73 && !PS2_LooksLikeGlitch()) {
		ud = (int16_t)0x80 - (int16_t)my_joystick.RJoy_UD;
		lr = (int16_t)0x80 - (int16_t)my_joystick.RJoy_LR;
		ll = (int16_t)0x80 - (int16_t)my_joystick.LJoy_LR;

		if (ud > -PS2_DEADZONE && ud < PS2_DEADZONE) ud = 0;
		if (lr > -PS2_DEADZONE && lr < PS2_DEADZONE) lr = 0;
		if (ll > -PS2_DEADZONE && ll < PS2_DEADZONE) ll = 0;

		R_Vel.TG_IX = (int16_t)(speed * ud);
		R_Vel.TG_IY = (int16_t)(speed * lr);
		R_Vel.TG_IW = (int16_t)(4 * speed * ll);

		/* L-stick click: slow down */
		if (my_joystick.btn1 & PS2_BT1_JOY_L) {
			btn_joyl_flag = 1;
		} else if (btn_joyl_flag) {
			if (speed > 2) {
				speed--;
			} else {
				speed = 2;
				ax_beep_ring = BEEP_SHORT;
			}
			btn_joyl_flag = 0;
		}

		/* R-stick click: speed up */
		if (my_joystick.btn1 & PS2_BT1_JOY_R) {
			btn_joyr_flag = 1;
		} else if (btn_joyr_flag) {
			if (speed < 9) {
				speed++;
			} else {
				speed = 9;
				ax_beep_ring = BEEP_SHORT;
			}
			btn_joyr_flag = 0;
		}
	} else {
		/* Digital / un-handshaked mode: hold zero so a button-only override
		 * (e.g. user mashed Start) brings the robot to a definitive stop
		 * instead of inheriting whatever target was there. */
		R_Vel.TG_IX = 0;
		R_Vel.TG_IY = 0;
		R_Vel.TG_IW = 0;
	}
}

	
/**
  * @??  ??  ?????????????
  * @??  ??  ??
  * @?????  ??
  */
void Robot_Task(void* parameter)
{	

	//????????????????��??????????
	static portTickType PreviousWakeTime;

	/* Sticky timer: refreshed every cycle the PS2 looks active. PS2 keeps the
	 * wheels until it has been quiet for PS2_RELEASE_MS, at which point Pi /
	 * APP / FN1 may take over again. */
	static portTickType ps2_last_active_tick = 0;
	static uint8_t      ps2_was_active       = 0;
	static portTickType robot_task_start_tick = 0;
	uint32_t            now_tick;
	uint8_t             ps2_warmup_done;
	uint8_t             ps2_active_now;
	uint8_t             ps2_glitch_now;
	uint8_t             ps2_override;
	uint8_t             pi_live;
	uint8_t             ctrl_flags;

	//??????????20ms??????????????? 
	const portTickType TimeIncrement = pdMS_TO_TICKS(20);
	
	//??????????? 
	PreviousWakeTime    = xTaskGetTickCount();
	robot_task_start_tick = PreviousWakeTime;
	
	while(1)
	{
		
		//??????????????20ms,??????50HZ
		vTaskDelayUntil(&PreviousWakeTime, TimeIncrement );
		
		//??????????????????????/???????
		AX_ULTRASONIC_Update();
		
		//Ultrasonic debug output (USART1 230400baud)
		printf("US F:%u R:%u\r\n", ax_ultrasonic_front_mm, ax_ultrasonic_rear_mm);

		now_tick = xTaskGetTickCount();

		/* --- Input arbitration ----------------------------------------------
		 * Priority (highest first):
		 *   1. PS2 pad (override) - any human touch wins instantly.
		 *   2. Pi  cmd_vel        - default driver while idle.
		 *   3. APP / FN1          - legacy fallbacks if no Pi link.
		 * The host is informed of the active source via UART4 CMD 0x12 each
		 * tick so it can pause its own velocity stream while the user is
		 * grabbing the wheel.
		 *
		 * Boot-time safety: PS2_IsActive() can spuriously fire while a cheap
		 * receiver is still negotiating with the controller (mode briefly
		 * reports 0x73 with junk stick values). We ignore PS2 for the first
		 * PS2_WARMUP_MS so the robot cannot launch itself on power-up.
		 */
		ps2_warmup_done = ((now_tick - robot_task_start_tick) >=
		                   pdMS_TO_TICKS(PS2_WARMUP_MS));

		ps2_active_now = (ps2_warmup_done && PS2_IsActive());
		ps2_glitch_now = PS2_LooksLikeGlitch();

		if (ps2_active_now) {
			ps2_last_active_tick = now_tick;
			ps2_was_active       = 1;
		}

		ps2_override = (ps2_was_active != 0) &&
		               ((now_tick - ps2_last_active_tick) < pdMS_TO_TICKS(PS2_RELEASE_MS));

		pi_live = (ax_uart4_pi_cmd_tick != 0) &&
		          ((now_tick - ax_uart4_pi_cmd_tick) < pdMS_TO_TICKS(PI_CMD_TIMEOUT_MS));

		ctrl_flags = 0;
		if (ps2_active_now)   ctrl_flags |= AX_CTRL_FLAG_PS2_ACTIVE;
		if (ps2_override)     ctrl_flags |= AX_CTRL_FLAG_PS2_OVERRIDE;
		if (ps2_warmup_done)  ctrl_flags |= AX_CTRL_FLAG_PS2_WARMUP;
		if (pi_live)          ctrl_flags |= AX_CTRL_FLAG_PI_LIVE;
		if (ps2_glitch_now)   ctrl_flags |= AX_CTRL_FLAG_PS2_GLITCH;

		if (ps2_override) {
			if (ps2_active_now) {
				/* User is currently touching the pad: drive from sticks. */
				Robot_ApplyPs2();
			} else {
				/* User has let go but we are still inside the 800 ms grace
				 * window. Brake immediately instead of letting Robot_ApplyPs2
				 * compute a near-zero (or in-deadzone) speed from a slightly
				 * miscentered stick - the latter is what makes a knock-off pad
				 * "drift forward" after release. The override latch is kept so
				 * the Pi cannot snatch the wheels for another PS2_RELEASE_MS,
				 * giving the user time to grab the stick again. */
				R_Vel.TG_IX = 0;
				R_Vel.TG_IY = 0;
				R_Vel.TG_IW = 0;
				ctrl_flags |= AX_CTRL_FLAG_PS2_HOLD0;
			}
			ax_robot_move_enable = 1;
			ax_ctrl_source       = AX_CTRL_SRC_PS2;
		} else {
			/* PS2 has finished its grace window; clear the latch so the next
			 * PI / APP frame is honored cleanly. */
			ps2_was_active = 0;

			if (pi_live) {
				R_Vel.TG_IX          = ax_uart4_pi_vx;
				R_Vel.TG_IY          = ax_uart4_pi_vy;
				R_Vel.TG_IW          = ax_uart4_pi_vw;
				ax_robot_move_enable = 1;
				ax_ctrl_source       = AX_CTRL_SRC_PI;
			} else if (ax_control_mode == CTL_APP && ax_robot_move_enable) {
				/* Bluetooth APP requires KEY-long-press to arm
				 * (ax_robot_move_enable = 1 from Key_Task) - matches the
				 * safety contract of the original firmware. The legacy FN1
				 * (gyro-locked auto-straight) branch is intentionally NOT
				 * dispatched here: AX_FUN_Ls1() unconditionally writes
				 * R_Vel.TG_IX = 200, which silently drives the robot
				 * forward whenever the Pi link drops for >300 ms. If you
				 * really want autonomous line-following, restore it under
				 * an explicit, audible-on-arm guard. */
				AX_CTL_App();
				ax_ctrl_source = AX_CTRL_SRC_APP;
			} else {
				/* Nobody's driving and no auto-mode is armed: brake. The
				 * software emergency-stop in AX_ROBOT_Kinematics() honours
				 * ax_robot_move_enable == 0 as well, so this acts as a
				 * second-line guarantee that the wheels stop. */
				R_Vel.TG_IX          = 0;
				R_Vel.TG_IY          = 0;
				R_Vel.TG_IW          = 0;
				ax_robot_move_enable = 0;
				ax_ctrl_source       = AX_CTRL_SRC_IDLE;
			}
		}

		AX_ROBOT_Kinematics();
		AX_UART4_SendPiTelemetry();
		AX_UART4_SendPiPS2();
		AX_UART4_SendPiCtrl(ax_ctrl_source, ctrl_flags,
		                    R_Vel.TG_IX, R_Vel.TG_IY, R_Vel.TG_IW);
	}
}


/**
  * @??  ??  ???????????
  * @??  ??  ??
  * @?????  ??
  */
void Trivia_Task(void* parameter)
{	
	//????????
	static uint16_t ax_bat_vol_cnt = 0; 
	
	while (1)
	{	
		
		/*****??????***********************************/
		
		//????????
	    R_Bat_Vol = AX_VIN_GetVol_X100();
		
		//????????????????
        //printf("@ %d  \r\n",R_Bat_Vol);		
		
		//????????40%
		if(R_Bat_Vol < VBAT_40P)  
		{
			//???????????
			AX_LED_Red_Toggle();
			
			//????????20%
			if(R_Bat_Vol < VBAT_20P)
			{
				//??????
				AX_LED_Red_On();
				
				//????????10%???????????????
				if(R_Bat_Vol < VBAT_10P) //990
				{
					//?????????
					ax_bat_vol_cnt++;
					
					//????10?��?????????
					if(ax_bat_vol_cnt > 10 )
					{
						//?????????????
						AX_LED_Green_Off();
						AX_LED_Red_On();
						
						//???????
						vTaskSuspend(Robot_Task_Handle);
						vTaskSuspend(Disp_Task_Handle);
						
						//???????????0
						AX_MOTOR_A_SetSpeed(0);
						AX_MOTOR_B_SetSpeed(0);  
						
						//???OLED???????????
						AX_OLED_ClearScreen();  //
						
						//?????????��???
						while(1)
						{	
							//??????????????
							AX_OLED_DispStr(0, 3, "     Low power      ", 0);	
							AX_OLED_DispStr(0, 5, "  Robot has stopped ", 0);	
							AX_BEEP_On();
							vTaskDelay(30);
							AX_BEEP_Off();
							
							vTaskDelay(1000);	
							AX_OLED_ClearScreen(); 
							vTaskDelay(1000);						
						}								
					}
				}
				else
				{
					ax_bat_vol_cnt = 0;
				}				
			}
		}
		else
		{
			//?????
			AX_LED_Red_Off();
		}	
		
		/*****?????????��???***********************************/
		if(ax_beep_ring != 0)
		{
			if(ax_beep_ring == BEEP_SHORT)
			{
				//?????????
				AX_BEEP_On();
				vTaskDelay(200); 
				AX_BEEP_Off();
				
				//????????????��
				ax_beep_ring = 0;
			}
			else //
			{
				//?????????
				AX_BEEP_On();
				vTaskDelay(1000); 
				AX_BEEP_Off();
				
				//????????????��
				ax_beep_ring = 0;				
			}
		}
		
		//LED????????
		AX_LED_Green_Toggle();	
		
		
        //???????500ms
		vTaskDelay(500); 
	}			
}

/**
  * @??  ??  ????????????
  * @??  ??  ??
  * @?????  ??
  */
void Key_Task(void* parameter)
{	
	uint8_t  i;
	//int16_t  temp;
	
	while (1)
	{		
		//???????
		if(AX_KEY_Scan() != 0)
		{
			//???????
			vTaskDelay(50);  
			
			//???????????
			if(AX_KEY_Scan() != 0)
			{
				//??????????
				for(i=0; i<200; i++)
				{

					vTaskDelay(50);
					
					if(AX_KEY_Scan() == 0)
					{
						break;
					}
					
					//???????3S????????????
					if(i == 60)
					{
						AX_BEEP_On();
						vTaskDelay(200);
						AX_BEEP_Off();
					}
				}

				//??????,��??1S
				if(i < 20)
				{
					//??????????????
					ax_robot_move_enable = 0;
				
					
					//?��???????
					ax_control_mode = CTL_FN1;
					
					//?????????
					AX_BEEP_On();
					vTaskDelay(50); 
					AX_BEEP_Off();	
				}
				
				//?��????,????3S??��??10S
				if(i>60 && i<200)
				{
					//????????????????
					ax_robot_move_enable = 1;
				}				
				
					
				//???????????10S
				if(i == 200)
				{
					//???????					
				}
			}
		}
		
		//???????
		vTaskDelay(50);    
	}			
}

/**
  * @??  ??  ????????????
  * @??  ??  ??
  * @?????  ??
  */
void Disp_Task(void* parameter)
{	

	while (1)
	{	
		//???	
		vTaskDelay(100); 
		
		//??3????��?????????
		if      (ax_control_mode == CTL_PS2)    AX_OLED_DispStr(30, 2, "PS2", 0);   //PS2???????
		else if (ax_control_mode == CTL_APP)    AX_OLED_DispStr(30, 2, "APP", 0);   //APP????
		else if (ax_control_mode == CTL_FN1)    AX_OLED_DispStr(30, 2, "FN1", 0);   //CCD?????	

		
		//??3?��??��????????
		AX_OLED_DispValue(90, 2, (R_Bat_Vol*0.1), 2, 1, 0);
		
		//??4?��????Z????????????
		AX_OLED_DispValue(30, 3, (ax_imu_gyro_data[2]), 6, 0, 0);	
		
		//??5?��??????????????(mm)
		if(ax_ultrasonic_front_mm < 0xFFFF)
			AX_OLED_DispValue(30, 4, ax_ultrasonic_front_mm, 4, 0, 0);
		else
			AX_OLED_DispStr(30, 4, "----", 0);
		
		if(ax_ultrasonic_rear_mm < 0xFFFF)
			AX_OLED_DispValue(90, 4, ax_ultrasonic_rear_mm, 4, 0, 0);
		else
			AX_OLED_DispStr(90, 4, "----", 0);
		
		vTaskDelay(100); 
		
		//??7,8?��?????????????
		AX_OLED_DispValue(30, 6, (R_Wheel_A.RT*100), 2, 2, 0);
		AX_OLED_DispValue(90, 6, (R_Wheel_B.RT*100), 2, 2, 0);
	}			
}

/**
  * @??  ??  PS2??????????
  * @??  ??  ??
  * @?????  ??
  */
void Ps2_Task(void* parameter)
{	

	//????????????????��??????????
	static portTickType PreviousWakeTime1;

	//??????????20ms??????????????? 
	const portTickType TimeIncrement1 = pdMS_TO_TICKS(20);
	
	//??????????? 
	PreviousWakeTime1 = xTaskGetTickCount();
	
	while(1)
	{
		
		//??????????????20ms,??????50HZ
		vTaskDelayUntil(&PreviousWakeTime1, TimeIncrement1 );
		
		//???PS2??????
		AX_PS2_ScanKey(&my_joystick);
		
		//????PS2????????
		if(ax_control_mode != CTL_PS2)
		{
			//?��??????PS2???????
			//START??????????????????????????PS2??????
			if((my_joystick.btn1 == PS2_BT1_START) && (my_joystick.LJoy_UD == 0x00))
			{
				//?��???PS2??
				ax_control_mode = CTL_PS2;	
				
				//?????????????
				ax_robot_move_enable = 1;

				//??��????????????
				ax_beep_ring = BEEP_SHORT;
			}
		}
		
//		//?????????
//		printf("MODE:%2x BTN1:%2x BTN2:%2x RJOY_LR:%2x RJOY_UD:%2x LJOY_LR:%2x LJOY_UD:%2x\r\n",
//		my_joystick.mode, my_joystick.btn1, my_joystick.btn2, 
//		my_joystick.RJoy_LR, my_joystick.RJoy_UD, my_joystick.LJoy_LR, my_joystick.LJoy_UD);
	}
}

/******************* (C) ??? 2023 XTARK **************************************/


/**			                                                    
		   ____                    _____ _______ _____       @���˴���
		  / __ \                  / ____|__   __|  __ \ 
		 | |  | |_ __   ___ _ __ | |       | |  | |__) |
		 | |  | | '_ \ / _ \ '_ \| |       | |  |  _  / 
		 | |__| | |_) |  __/ | | | |____   | |  | | \ \ 
		  \____/| .__/ \___|_| |_|\_____|  |_|  |_|  \_\
				| |                                     
				|_|                OpenCTR   �����˿�����
									 
  ****************************************************************************** 
  *           
  * ��Ȩ���У� @���˴���  ��Ȩ���У�����ؾ�
  * ��˾��վ�� www.xtark.cn   www.tarkbot.com
  * �Ա����̣� https://xtark.taobao.com  
  * ����΢�ţ� ���˴��£���ע���ںţ���ȡ���¸�����Ѷ��
  *      
  ******************************************************************************
  * @��  ��  Musk Han@XTARK
  * @��  ��  TTL����ͨ��
  *
  ******************************************************************************
  */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __AX_UART4_H
#define __AX_UART4_H

/* Includes ------------------------------------------------------------------*/	 
#include "stm32f10x.h"

//OpenCTR�ӿں���
void    AX_UART4_Init(uint32_t baud);  //��չ���ڳ�ʼ��
uint8_t AX_UART4_GetData(uint8_t *pbuf);
void    AX_UART4_SendPacket(uint8_t *pbuf, uint8_t len, uint8_t num);  //�������ݣ�X-ProtocolЭ�飩

/* Telemetry / PS2 forwarding to client/serial_comm.py (UART4 PC10/PC11) */
void    AX_UART4_SendPiTelemetry(void);
void    AX_UART4_SendPiPS2(void);

/* CMD 0x12, 8-byte arbitration / ctrl-state diagnostic frame.
 * Layout (big-endian):
 *   [0]    source            (one of AX_CTRL_SRC_*)
 *   [1]    flags             (bitmask of AX_CTRL_FLAG_*)
 *   [2-3]  target vx (mm/s)  (int16, mirrors R_Vel.TG_IX)
 *   [4-5]  target vy (mm/s)  (int16)
 *   [6-7]  target vw (mrad/s)(int16)
 */
void    AX_UART4_SendPiCtrl(uint8_t source, uint8_t flags,
                            int16_t tg_vx, int16_t tg_vy, int16_t tg_vw);

/* Control source IDs reported to the host via CMD 0x12. */
#define AX_CTRL_SRC_IDLE   0u   /* nothing is driving the wheels */
#define AX_CTRL_SRC_PS2    1u   /* PS2 joystick has overridden everything */
#define AX_CTRL_SRC_PI     2u   /* Pi/ROS cmd_vel via 0x50 */
#define AX_CTRL_SRC_APP    3u   /* Bluetooth APP via USART2 */
#define AX_CTRL_SRC_FN1    4u   /* line-following / autonomous fallback */

/* Diagnostic flags packed into the 0x12 frame's second byte. */
#define AX_CTRL_FLAG_PS2_ACTIVE   0x01u  /* PS2_IsActive() returned 1 this tick */
#define AX_CTRL_FLAG_PS2_OVERRIDE 0x02u  /* PS2 is currently overriding (in 800ms grace) */
#define AX_CTRL_FLAG_PS2_WARMUP   0x04u  /* warm-up window elapsed (PS2 allowed to grab) */
#define AX_CTRL_FLAG_PI_LIVE      0x08u  /* Pi cmd_vel within 300ms freshness */
#define AX_CTRL_FLAG_PS2_GLITCH   0x10u  /* PS2_LooksLikeGlitch() detected this tick */
#define AX_CTRL_FLAG_PS2_HOLD0    0x20u  /* PS2 in override but sticks released -> braking */

extern volatile uint32_t ax_uart4_pi_cmd_tick;
extern volatile int16_t  ax_uart4_pi_vx;
extern volatile int16_t  ax_uart4_pi_vy;
extern volatile int16_t  ax_uart4_pi_vw;

#endif 

/******************* (C) ��Ȩ 2023 XTARK **************************************/

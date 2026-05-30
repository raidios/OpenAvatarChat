/**
 ******************************************************************************
 * TTL UART4 (PC10 TX / PC11 RX) ?? X-Protocol for Raspberry Pi host.
 * Frame: AA 55 | LEN | CMD | payload | checksum (sum of all prior bytes & 0xFF)
 * Matches OpenAvatarChat client/serial_comm.py (115200 8N1).
 ******************************************************************************
 */

#include "ax_uart4.h"
#include <stdio.h>
#include "ax_robot.h"
#include "ax_mpu6050.h"
#include "FreeRTOS.h"
#include "task.h"

#define ID_CPR2ROS_DATA   0x10u
#define ID_CPR2ROS_PS2    0x11u
#define ID_CPR2ROS_CTL    0x12u
#define ID_ROS2CRP_VEL    0x50u
#define ID_ROS2CRP_IMU    0x51u
#define ID_ROS2CRP_AKM    0x5Fu
#define UART4_RX_BUF_SZ   60u

static uint8_t uart4_rx_ok = 0;
static uint8_t uart4_rx_con = 0;
static uint8_t uart4_rx_checksum;
static uint8_t uart4_rx_buf[UART4_RX_BUF_SZ];
static uint8_t uart4_tx_buf[60];

volatile uint32_t ax_uart4_pi_cmd_tick = 0;
volatile int16_t  ax_uart4_pi_vx = 0;
volatile int16_t  ax_uart4_pi_vy = 0;
volatile int16_t  ax_uart4_pi_vw = 0;

void AX_UART4_Init(uint32_t baud)
{
	GPIO_InitTypeDef GPIO_InitStructure;
	USART_InitTypeDef USART_InitStructure;
	NVIC_InitTypeDef NVIC_InitStructure;

	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOC, ENABLE);

	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOC, &GPIO_InitStructure);

	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_11;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
	GPIO_Init(GPIOC, &GPIO_InitStructure);

	RCC_APB1PeriphClockCmd(RCC_APB1Periph_UART4, ENABLE);

	USART_InitStructure.USART_BaudRate = baud;
	USART_InitStructure.USART_WordLength = USART_WordLength_8b;
	USART_InitStructure.USART_StopBits = USART_StopBits_1;
	USART_InitStructure.USART_Parity = USART_Parity_No;
	USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
	USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
	USART_Init(UART4, &USART_InitStructure);

	NVIC_InitStructure.NVIC_IRQChannel = UART4_IRQn;
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 2;
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 1;
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
	NVIC_Init(&NVIC_InitStructure);

	USART_ITConfig(UART4, USART_IT_RXNE, ENABLE);
	USART_Cmd(UART4, ENABLE);
}

void UART4_IRQHandler(void)
{
	uint8_t Res;

	if (USART_GetITStatus(UART4, USART_IT_RXNE) == RESET)
		return;

	Res = (uint8_t)USART_ReceiveData(UART4);

	if (uart4_rx_con < 3) {
		if (uart4_rx_con == 0) {
			if (Res == 0xAA) {
				uart4_rx_buf[0] = Res;
				uart4_rx_con = 1;
			}
		} else if (uart4_rx_con == 1) {
			if (Res == 0x55) {
				uart4_rx_buf[1] = Res;
				uart4_rx_con = 2;
			} else {
				uart4_rx_con = 0;
			}
		} else {
			if (Res < 5u || Res > UART4_RX_BUF_SZ) {
				uart4_rx_con = 0;
			} else {
				uart4_rx_buf[2] = Res;
				uart4_rx_con = 3;
				uart4_rx_checksum = (uint8_t)(0xAA + 0x55 + Res);
			}
		}
	} else {
		if (uart4_rx_con < (uart4_rx_buf[2] - 1u)) {
			if (uart4_rx_con >= UART4_RX_BUF_SZ) {
				uart4_rx_con = 0;
			} else {
				uart4_rx_buf[uart4_rx_con] = Res;
				uart4_rx_con++;
				uart4_rx_checksum = (uint8_t)(uart4_rx_checksum + Res);
			}
		} else {
			uart4_rx_con = 0;
			if (Res == uart4_rx_checksum) {
				uint8_t flen = uart4_rx_buf[2];
				uint8_t cmd  = uart4_rx_buf[3];

				if (cmd == ID_ROS2CRP_VEL && flen >= 11u) {
					ax_uart4_pi_vx = (int16_t)((uart4_rx_buf[4] << 8) | uart4_rx_buf[5]);
					ax_uart4_pi_vy = (int16_t)((uart4_rx_buf[6] << 8) | uart4_rx_buf[7]);
					ax_uart4_pi_vw = (int16_t)((uart4_rx_buf[8] << 8) | uart4_rx_buf[9]);
					/* Older FreeRTOS: xTaskGetTickCountFromISR(void) — no pxHigherPriorityTaskWoken */
					ax_uart4_pi_cmd_tick = xTaskGetTickCountFromISR();
				}
				/* TODO: ID_ROS2CRP_IMU / ID_ROS2CRP_AKM if needed */

				uart4_rx_ok = 1;
			}
		}
	}

	USART_ITConfig(UART4, USART_IT_RXNE, ENABLE);
}

uint8_t AX_UART4_GetData(uint8_t *pbuf)
{
	uint8_t cnt, i;

	if (uart4_rx_ok != 0) {
		cnt = (uint8_t)(uart4_rx_buf[2] - 4u);
		for (i = 0; i < cnt; i++)
			*(pbuf + i) = uart4_rx_buf[3 + i];
		uart4_rx_ok = 0;
		return cnt;
	}
	return 0;
}

void AX_UART4_SendPacket(uint8_t *pbuf, uint8_t len, uint8_t num)
{
	uint8_t i, cnt;
	uint8_t tx_checksum = 0;

	if (len > 50u)
		return;

	uart4_tx_buf[0] = 0xAA;
	uart4_tx_buf[1] = 0x55;
	uart4_tx_buf[2] = (uint8_t)(len + 5u);
	uart4_tx_buf[3] = num;

	for (i = 0; i < len; i++)
		uart4_tx_buf[4 + i] = *(pbuf + i);

	cnt = (uint8_t)(4u + len);
	for (i = 0; i < cnt; i++)
		tx_checksum = (uint8_t)(tx_checksum + uart4_tx_buf[i]);
	uart4_tx_buf[cnt] = tx_checksum;

	cnt = (uint8_t)(5u + len);
	for (i = 0; i < cnt; i++) {
		USART_SendData(UART4, uart4_tx_buf[i]);
		while (USART_GetFlagStatus(UART4, USART_FLAG_TC) == RESET)
			;
	}
}

/**
 * @brief 50 Hz telemetry ?? same 20-byte big-endian layout as legacy robot.c / Python parser.
 */
void AX_UART4_SendPiTelemetry(void)
{
	int16_t acc_raw[3], gyro_raw[3];
	int16_t acc_x, acc_y, acc_z, gx, gy, gz;
	uint8_t comdata[20];

	AX_MPU6050_GetAccData(acc_raw);
	acc_x = (int16_t)(-acc_raw[1]);
	acc_y = acc_raw[0];
	acc_z = acc_raw[2];

	AX_MPU6050_GetGyroData(gyro_raw);
	gyro_raw[0] += ax_imu_gyro_offset[0];
	gyro_raw[1] += ax_imu_gyro_offset[1];
	gyro_raw[2] += ax_imu_gyro_offset[2];
	gx = (int16_t)(-gyro_raw[1]);
	gy = gyro_raw[0];
	gz = gyro_raw[2];

	comdata[0]  = (uint8_t)(acc_x >> 8);
	comdata[1]  = (uint8_t)acc_x;
	comdata[2]  = (uint8_t)(acc_y >> 8);
	comdata[3]  = (uint8_t)acc_y;
	comdata[4]  = (uint8_t)(acc_z >> 8);
	comdata[5]  = (uint8_t)acc_z;
	comdata[6]  = (uint8_t)(gx >> 8);
	comdata[7]  = (uint8_t)gx;
	comdata[8]  = (uint8_t)(gy >> 8);
	comdata[9]  = (uint8_t)gy;
	comdata[10] = (uint8_t)(gz >> 8);
	comdata[11] = (uint8_t)gz;
	comdata[12] = (uint8_t)(R_Vel.RT_IX >> 8);
	comdata[13] = (uint8_t)R_Vel.RT_IX;
	comdata[14] = (uint8_t)(R_Vel.RT_IY >> 8);
	comdata[15] = (uint8_t)R_Vel.RT_IY;
	comdata[16] = (uint8_t)(R_Vel.RT_IW >> 8);
	comdata[17] = (uint8_t)R_Vel.RT_IW;
	comdata[18] = (uint8_t)(R_Bat_Vol >> 8);
	comdata[19] = (uint8_t)R_Bat_Vol;

	AX_UART4_SendPacket(comdata, 20, ID_CPR2ROS_DATA);
}

/**
 * @brief Send PS2 joystick state to Pi (CMD 0x11, 7 bytes).
 *        Layout: mode | btn1 | btn2 | RJoy_LR | RJoy_UD | LJoy_LR | LJoy_UD
 */
void AX_UART4_SendPiPS2(void)
{
	uint8_t buf[7];
	buf[0] = my_joystick.mode;
	buf[1] = my_joystick.btn1;
	buf[2] = my_joystick.btn2;
	buf[3] = my_joystick.RJoy_LR;
	buf[4] = my_joystick.RJoy_UD;
	buf[5] = my_joystick.LJoy_LR;
	buf[6] = my_joystick.LJoy_UD;
	AX_UART4_SendPacket(buf, 7, ID_CPR2ROS_PS2);
}

/**
 * @brief Tell the host which input source is currently driving the wheels and
 *        expose enough arbitration state to debug "robot won't stop"-class
 *        issues without an MCU debug serial. CMD 0x12, 8 bytes.
 *
 *        Layout (big-endian):
 *          [0]    source       (AX_CTRL_SRC_*)
 *          [1]    flags        (AX_CTRL_FLAG_*)
 *          [2-3]  target vx    (int16, mm/s)
 *          [4-5]  target vy    (int16, mm/s)
 *          [6-7]  target vw    (int16, mrad/s)
 */
void AX_UART4_SendPiCtrl(uint8_t source, uint8_t flags,
                         int16_t tg_vx, int16_t tg_vy, int16_t tg_vw)
{
	uint8_t buf[8];
	buf[0] = source;
	buf[1] = flags;
	buf[2] = (uint8_t)(tg_vx >> 8);
	buf[3] = (uint8_t)tg_vx;
	buf[4] = (uint8_t)(tg_vy >> 8);
	buf[5] = (uint8_t)tg_vy;
	buf[6] = (uint8_t)(tg_vw >> 8);
	buf[7] = (uint8_t)tg_vw;
	AX_UART4_SendPacket(buf, 8, ID_CPR2ROS_CTL);
}

#!/usr/bin/python
# -*- coding: utf-8 -*-

# Programa simples com camera webcam e opencv

import cv2
import os, sys, os.path
import numpy as np
import math

# Ciano
image_lower_hsv1 = np.array([70, 50, 80])
image_upper_hsv1 = np.array([100, 255, 255])

# Vermelho
image_lower_hsv2 = np.array([0, 80, 60])
image_upper_hsv2 = np.array([10, 255, 255])


def filtro_de_cor(img_bgr, low_hsv, high_hsv):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img, low_hsv, high_hsv)
    return mask


def mascara_or(mask1, mask2):
    mask = cv2.bitwise_or(mask1, mask2)
    return mask


def desenha_cruz(img, cX, cY, size, color):
    cv2.line(img, (cX - size, cY), (cX + size, cY), color, 5)
    cv2.line(img, (cX, cY - size), (cX, cY + size), color, 5)


def escreve_texto(img, text, origem, color):

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(text), origem, font, 1, color, 2, cv2.LINE_AA)


def image_da_webcam(img):

    mask_hsv1 = filtro_de_cor(img, image_lower_hsv1, image_upper_hsv1)
    mask_hsv2 = filtro_de_cor(img, image_lower_hsv2, image_upper_hsv2)

    mask_hsv = mascara_or(mask_hsv1, mask_hsv2)

    contornos, _ = cv2.findContours(mask_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask_rgb = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2RGB)
    contornos_img = mask_rgb.copy()

    maior = None
    maior_area = 0
    segunda_maior = None
    segunda_maior_area = 0
    for c in contornos:
        area = cv2.contourArea(c)
        if area > maior_area:
            if maior_area > segunda_maior_area:
                segunda_maior_area = maior_area
                segunda_maior = maior
            maior_area = area
            maior = c
        elif area > segunda_maior_area:
            segunda_maior_area = area
            segunda_maior = c

    M = cv2.moments(maior)
    M2 = cv2.moments(segunda_maior)

    # Verifica se existe alguma para calcular, se sim calcula e exibe no display
    if M["m00"] != 0 and M2["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cX2 = int(M2["m10"] / M2["m00"])
        cY2 = int(M2["m01"] / M2["m00"])

        cv2.drawContours(contornos_img, [maior], -1, [255, 0, 0], 5)
        cv2.drawContours(contornos_img, [segunda_maior], -1, [255, 0, 0], 5)

        # faz a cruz no centro de massa
        desenha_cruz(contornos_img, cX, cY, 20, (0, 0, 255))
        desenha_cruz(contornos_img, cX2, cY2, 20, (0, 0, 255))

        # Para escrever vamos definir uma fonte
        text1 = cX, cY
        text2 = cX2, cY2

        escreve_texto(contornos_img, text1, (400, cY), (0, 0, 255))
        escreve_texto(contornos_img, text2, (400, cY2), (0, 0, 255))

        cv2.line(contornos_img, (cX, cY), (cX2, cY2), (0, 0, 255), 5)

        rad = math.atan2(cY - cY2, cX - cY2)
        graus = round(math.degrees(rad))

        text = 'Angulo formado: ' + str(int(graus))
        origem = (300, contornos_img.shape[0] - 50)

        escreve_texto(contornos_img, text, origem, (0, 0, 255))

    else:
        # se não existe nada para segmentar
        # Para escrever vamos definir uma fonte
        text1 = 'nao tem nada'
        escreve_texto(contornos_img, text1, (100, 100), (0, 0, 255))

    return contornos_img


cv2.namedWindow("preview")
# define a entrada de video para webcam
vc = cv2.VideoCapture('http://192.168.1.149:8080/video')

# configura o tamanho da janela
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:

    img = image_da_webcam(frame)  # passa o frame para a função imagem_da_webcam e recebe em img imagem tratada

    cv2.imshow("original", frame)
    cv2.imshow("preview", img)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()
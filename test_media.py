
import time
import pygame
import threading

# 初始化声音引擎
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("./media/fastdown.ogg")  # 确保文件存在[9,10](@ref)
is_playing = False  # 声音播放状态标志

def play_alert():
    """非阻塞播放声音（独立线程）"""
    global is_playing
    if not is_playing:
        is_playing = True
        alert_sound.play()
        # 播放结束后自动更新状态
        threading.Thread(target=wait_for_sound_end).start()

def stop_alert():
    """停止声音并重置状态"""
    global is_playing
    if is_playing:
        alert_sound.stop()
        is_playing = False

def wait_for_sound_end():
    """监听声音结束并更新状态"""
    global is_playing
    while pygame.mixer.get_busy():
        time.sleep(0.1)
    is_playing = False


if __name__ == '__main__':
    play_alert()
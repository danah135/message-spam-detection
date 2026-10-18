import asyncio
import os
import joblib
from telegram import Bot
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

BOT_TOKEN = os.environ.get("BOT_TOKEN")

if not BOT_TOKEN:
    print(" خطأ: BOT_TOKEN غير مضبوط!")
    exit()

PORT = int(os.environ.get("PORT", 10000))

model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

print(" تم تحميل النموذج!")
print(" البوت جاهز - أرسل له أي رسالة لتصنيفها\n")


def classify_message(message):
    msg_vectorized = vectorizer.transform([message])
    prediction = model.predict(msg_vectorized)[0]
    if prediction == 'spam':
        return " هذه الرسالة مصنفة كـ: احتيالية (SPAM)\n\n نصيحة: لا تضغطي على أي رابط ولا تشاركي معلومات شخصية."
    else:
        return " هذه الرسالة مصنفة كـ: حقيقية (HAM)"


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'OK')

    def log_message(self, format, *args):
        pass


def run_http_server():
    server = HTTPServer(('0.0.0.0', PORT), HealthCheckHandler)
    print(f" HTTP Server running on port {PORT}")
    server.serve_forever()


async def run_bot():
    my_bot = Bot(token=BOT_TOKEN)

    try:
        me = await my_bot.get_me()
        print(f" البوت متصل باسم: {me.username}")
    except Exception as e:
        print(f" خطأ في الاتصال: {e}")
        return

    last_update_id = None

    while True:
        try:
            updates = await my_bot.get_updates(offset=last_update_id, timeout=10)

            if not updates:
                continue

            for update in updates:
                last_update_id = update.update_id + 1

                if not update.message:
                    continue
                if not update.message.text:
                    continue

                user_message = update.message.text
                chat_id = update.message.chat.id

                print(f" رسالة من {update.message.from_user.first_name}: '{user_message[:50]}...'")

                result = classify_message(user_message)

                try:
                    await my_bot.send_message(chat_id=chat_id, text=result)
                    print(f"→ {' SPAM' if 'احتيالية' in result else ' HAM'} | تم الإرسال \n")
                except Exception as send_error:
                    print(f"→ فشل إرسال الرد: {send_error}\n")

        except Exception as e:
            if "timed out" not in str(e):
                print(f" خطأ في الاستقبال: {e}")
            await asyncio.sleep(5)


if __name__ == "__main__":
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    asyncio.run(run_bot())

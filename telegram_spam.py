import asyncio
import os
import joblib
from telegram import Bot
BOT_TOKEN = os.environ.get("BOT_TOKEN")
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
print("تم تحميل النموذج")
print("البوت جاهز - أرسل له أي رسالة لتصنيفها")
print("Ctrl + C لإيقاف البوت")


def classify_message(message):
    msg_vectorized = vectorizer.transform([message])
    prediction = model.predict(msg_vectorized)[0]
    if prediction == 'spam':
        return "⚠️ تحذير: هذه الرسالة مصنفة كـ: احتيالية (SPAM)"
    else:
        return "✅ هذه الرسالة مصنفة كـ: حقيقية (HAM)"

async def run_bot():
    my_bot = Bot(token=BOT_TOKEN)
    try:
            me = await my_bot.get_me()
            print(f"البوت متصل باسم: {me.username}")
    except Exception as e:
            print(f"❌ خطأ في الاتصال Token! تأكد من صحة الـ ⚠️ {e}")
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

             print(f"رسالة من {update.message.from_user.first_name}: {user_message[:50]}...")
            
             result = classify_message(user_message)
            
             try:
                await my_bot.send_message(chat_id=chat_id, text=result)
                print(f"تم الإرسال ➔ {result}\n")
             except Exception as send_error:
                print(f"فشل إرسال الرد: {send_error}\n")

        except Exception as e:
         if "timed out" not in str(e):
            print(f"⚠️ خطأ في الاستقبال: {e}")
            await asyncio.sleep(5)
asyncio.run(run_bot())

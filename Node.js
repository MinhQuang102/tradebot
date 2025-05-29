const express = require('express');
const app = express();
const port = process.env.PORT || 8080;

// Endpoint kiểm tra (tùy chọn)
app.get('/health', (req, res) => {
  res.send('OK');
});

// Khởi động bot Telegram
const TelegramBot = require('node-telegram-bot-api');
const token = '<your-bot-token>'; // Thay bằng token bot của bạn
const bot = new TelegramBot(token, { polling: true });

bot.on('message', (msg) => {
  const chatId = msg.chat.id;
  bot.sendMessage(chatId, 'Bot đang hoạt động!');
});

// Chạy server
app.listen(port, '0.0.0.0', () => {
  console.log(`Bot đang chạy trên cổng ${port}`);
});
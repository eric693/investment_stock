# Stock Monitor — 台美股終極 SOP 監控系統

終極 SOP 五大條件即時監控，含 LINE Bot 通知，可一鍵部署至 Render。

---

## 功能

| 模組 | 說明 |
|------|------|
| 股票看板 | 0050、QQQ、00631L、QLD 即時報價、日漲跌、200MA、50MA |
| 年線圖表 | 收盤價 + 200MA(年線) + 50MA 疊加圖，可切換標的 |
| SOP 五大條件 | 自動判斷狀態，顯示 正常 / 留意 / 警報 |
| 微笑佈局 | 0050 各買點（-8% ~ -30%）觸及狀況 |
| LINE 通知 | SOP 條件觸發時自動推播，或手動測試 |
| 自動更新 | 前端每 5 分鐘自動 fetch，後端 cache 300 秒 |

---

## SOP 邏輯

### 01 建軍配置
40% 原型(0050) + 40% 正2(00631L) + 20% 現金。每年底再平衡。

### 02 雙核雷達（警報條件）
- 台股看 0050 年線位置
- **QQQ 跌破年線 -10%** → 台股正2 無條件強制清倉

### 03 撤退機制（警報條件）
- 0050 **跌破年線 -3% 連續 3 天**，立刻出清正2轉備戰現金
- 或 0050 **單日跌幅 -5%**，同上

### 04 微笑佈局（線下買入）
- 線下只買 0050 原型
- 左側 -8%/-10%/-15%/-20%/-25%/-30% 各加碼 5%
- 右側（反彈後）各 2%
- 跌破 -30% 進入冬眠

### 05 反攻號角（買入條件）
- 0050 **站回年線 +3% 連續 3 天**
- 或 **單日 +5%**
- 底部原型 + 所有備戰現金，全數壓正2

---

## 本地執行

```bash
# 安裝套件
pip install -r requirements.txt

# 啟動
python app.py
# 瀏覽 http://localhost:5000
```

---

## 環境變數

| 變數 | 必填 | 說明 |
|------|------|------|
| `GAS_WEBHOOK_URL` | 否 | Google Apps Script 通知 URL（已預設） |
| `LINE_CHANNEL_ACCESS_TOKEN` | 否 | LINE Messaging API Token（直接推播用） |
| `LINE_USER_ID` | 否 | 接收通知的 LINE user ID / group ID |
| `CACHE_SECONDS` | 否 | 資料快取秒數，預設 300 |
| `PORT` | 否 | 伺服器埠號，Render 自動設定 |

---

## Google Apps Script 設定（LINE 通知）

如果您使用 GAS 轉發 LINE 通知，GAS 腳本需支援 POST，範例：

```javascript
function doPost(e) {
  const message = JSON.parse(e.postData.contents).message;
  const token = 'YOUR_LINE_NOTIFY_TOKEN';
  UrlFetchApp.fetch('https://notify-api.line.me/api/notify', {
    method: 'post',
    headers: { 'Authorization': 'Bearer ' + token },
    payload: 'message=' + encodeURIComponent(message)
  });
  return ContentService.createTextOutput('OK');
}
```

---

## 部署至 Render

1. **Push 到 GitHub**：建立 repo，push 所有檔案。

2. **Render 新建 Web Service**：
   - 選擇 GitHub repo
   - Runtime 選 **Python 3**，或使用 Dockerfile
   - Build Command：`pip install -r requirements.txt`
   - Start Command：`gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`

3. **設定環境變數**（Render Dashboard → Environment）：
   - `LINE_CHANNEL_ACCESS_TOKEN`
   - `LINE_USER_ID`
   - `GAS_WEBHOOK_URL`（可保留預設值）

4. 點擊 **Deploy**，等待部署完成後即可使用。

> Render 免費方案每 15 分鐘無請求會 spin down，建議使用付費方案或設定 UptimeRobot 定期 ping。

---

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/` | 儀表板首頁 |
| GET | `/api/dashboard` | 完整資料 JSON |
| POST | `/api/refresh` | 強制清除快取重新抓取 |
| POST | `/api/notify` | 手動發送 LINE 通知 |
| POST | `/webhook` | LINE Bot webhook（選用） |

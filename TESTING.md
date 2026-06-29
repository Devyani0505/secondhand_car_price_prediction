# AutoValue Testing Guide

This project is a Flask-based used-car valuation app with three main user flows:

- Car valuation
- EMI calculation
- Dealer comparison

## Manual Testing

Run the app:

```powershell
cd C:\Users\shri\OneDrive\Desktop\Final\car_price_prediction
..\venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:8000`.

### 1. Navigation and page load

- Open `/`, `/valuation`, `/emi`, and `/dealers`.
- Confirm each page loads without a server error.
- Confirm navigation links work from every page.
- Confirm the theme toggle changes the theme and keeps the UI readable.
- Confirm the layout is usable on desktop and mobile widths.

### 2. Valuation flow

Use a realistic car profile such as:

- Make: `Maruti`
- Model: `Alto`
- Variant: `VXI`
- City: `Hyderabad`
- Year: `2018`
- KMS: `40000`
- Mileage: `18`
- Owners: `1`
- Fuel: `Petrol`
- Transmission: `Manual`
- Color: `White`

Check these cases:

- Submit a fully valid form and confirm a price, range, and trend note are shown.
- Confirm the result card becomes visible only after a successful prediction.
- Confirm clicking `Check EMI options` opens `/emi?price=<predicted_price>`.
- Confirm valuation history stores recent successful predictions.
- Confirm `Use again` refills the form with the saved values.
- Wait more than 5 minutes and confirm old valuation history disappears.

Validation cases:

- Leave one required field empty and confirm the browser shows an error alert.
- Enter year below `1990` or above `2026` and confirm validation blocks submission.
- Enter `0` or negative values for kilometers, mileage, or owners and confirm validation blocks submission.

Pricing behavior checks:

- Run the same car with `market_trend_index = 1.00`, then `1.20`, and confirm the second result is higher.
- Change condition values from mostly `Good` to mostly `Poor` and confirm the price decreases.
- Increase owners and kilometers sharply and confirm the price decreases.
- Test an old, high-kilometer car and confirm the price does not become negative or blank.

### 3. EMI calculator

Check these cases:

- Open `/emi?price=<some_price>` from the valuation page and confirm the loan amount auto-fills.
- Enter a valid amount, rate, and years, then confirm EMI, months, total interest, and total payable appear.
- Confirm the doughnut chart renders after calculation.
- Change rate or tenure and confirm the EMI output changes.

Validation cases:

- Leave amount empty and confirm the app shows `Please enter valid loan details`.
- Enter `0` for rate or tenure and confirm the app blocks calculation.

### 4. Dealer comparison

Check these cases:

- Open `/dealers` and confirm all dealer cards render.
- Enter make and model and confirm each `Search This Car` link updates.
- Click `Compare On All Dealers` and confirm the first dealer search opens.
- Confirm the summary text reflects the current query.
- Click `Fill Example` and confirm sample values are inserted when the form is blank.

Validation cases:

- Click `Compare On All Dealers` with all fields empty and confirm the page asks for at least a make or model.
- Clear and re-enter different values and confirm updated links use the new query.

### 5. Backend API checks

You can also test the APIs directly with Postman or curl-like tools.

`POST /predict`

- Valid payload should return `success: true` with `predicted_price`, `price_low`, `price_high`, and trend metadata.
- Missing required fields should return `success: false`.
- Invalid year, kilometers, mileage, or owners should return `success: false`.

`POST /dealer-prices`

- Sending neither make nor model should return `success: false`.
- Sending at least one should return `success: true`, `query`, `found_count`, and `results`.

## Automated Testing

This repo now includes backend tests in `tests/test_app.py`.

Run them with:

```powershell
cd C:\Users\shri\OneDrive\Desktop\Final\car_price_prediction
..\venv\Scripts\python.exe -m unittest discover -s tests -v
```

### What the automated tests cover

- Page routes return HTTP 200.
- `/predict` fails cleanly when the model pipeline is missing.
- `/predict` rejects missing required fields.
- `/predict` rejects invalid year and non-positive kilometers.
- `/predict` returns a successful valuation response for a valid payload.
- `/predict` supports both manual market trend override and auto market trend mode.
- `/dealer-prices` validates required input.
- `/dealer-prices` returns normalized dealer results when the fetch layer succeeds.

### What is not automated yet

- Frontend DOM behavior in the browser
- Local storage history expiry behavior
- Theme toggle persistence
- EMI chart rendering
- Real network scraping against external dealer websites

Those are still best covered through manual testing or a future browser automation setup.

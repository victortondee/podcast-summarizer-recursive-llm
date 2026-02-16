const buyCreditsForm = document.getElementById('buyCreditsForm');
const creditAmountInput = document.getElementById('creditAmount');
const purchasePlanSelect = document.getElementById('purchasePlan');
const paymentMethodSelect = document.getElementById('paymentMethod');
const buyMessage = document.getElementById('buyMessage');
const confirmBuyBtn = document.getElementById('confirmBuyBtn');
const backToAppBtn = document.getElementById('backToAppBtn');

init();

async function init() {
    try {
        const response = await fetch('/auth/me', { credentials: 'include' });
        const data = await response.json();
        if (!response.ok || !data.authenticated || !data.user) {
            window.location.href = '/';
            return;
        }
    } catch (_) {
        window.location.href = '/';
    }
}

buyCreditsForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const credits = parseInt(creditAmountInput.value, 10);
    if (!Number.isFinite(credits) || credits <= 0) {
        setMessage('Please enter a valid credit amount.', true);
        return;
    }

    confirmBuyBtn.disabled = true;
    confirmBuyBtn.textContent = 'Processing...';

    try {
        const response = await fetch('/credits/purchase', {
            method: 'POST',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                credits,
                plan: purchasePlanSelect.value,
                payment_method: paymentMethodSelect.value,
            }),
        });
        const data = await response.json();
        if (!response.ok || !data.ok || !data.user) {
            throw new Error(data.detail || 'Purchase failed.');
        }

        setMessage(`Success. Added ${data.purchased_credits} credits.`, false);
    } catch (error) {
        setMessage(error.message || 'Purchase failed.', true);
    } finally {
        confirmBuyBtn.disabled = false;
        confirmBuyBtn.textContent = 'Continue to Checkout';
    }
});

backToAppBtn.addEventListener('click', () => {
    window.location.href = '/';
});

function setMessage(text, isError) {
    buyMessage.textContent = text;
    buyMessage.classList.remove('hidden');
    buyMessage.style.color = isError ? '#EF4444' : '#10B981';
}

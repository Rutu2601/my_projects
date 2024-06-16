document.addEventListener('DOMContentLoaded', () => {
    console.log('Netflix Clone Loaded');
});

document.getElementById('login-form').addEventListener('submit', function(e) {
    e.preventDefault();
    var email = document.getElementById('email').value;
    var password = document.getElementById('password').value;

    if (validateEmail(email) && validatePassword(password)) {
        alert('Login successful!');
        // Proceed with login or form submission
    } else {
        alert('Please enter valid credentials.');
    }
});

function validateEmail(email) {
    var re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

function validatePassword(password) {
    // Add password validation logic if needed
    return password.length > 0;
}

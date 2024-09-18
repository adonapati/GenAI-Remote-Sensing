import 'dart:math';
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:mailer/mailer.dart';
import 'package:mailer/smtp_server.dart';
import 'package:remote_sensing/HomePage.dart';
import 'package:remote_sensing/config.dart';

class SignupPage extends StatefulWidget {
  const SignupPage({Key? key}) : super(key: key);

  @override
  _SignupPageState createState() => _SignupPageState();
}

class _SignupPageState extends State<SignupPage> {
  final PageController _pageController = PageController(initialPage: 0);
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _phoneController = TextEditingController();
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _addressController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _confirmPasswordController = TextEditingController();
  final List<TextEditingController> _otpControllers =
      List.generate(6, (index) => TextEditingController());

  int _currentStep = 0;
  String _emailOtp = '';

  final String nexmoApiKey = Config.nexmoApiKey;
  final String nexmoApiSecret = Config.nexmoApiSecret;
  final String nexmoFromNumber = 'Nexmo';

  String _generatedOtp = '';

  void nextStep() {
    if (_currentStep < 4) {
      _pageController.nextPage(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeIn,
      );
      setState(() {
        _currentStep++;
      });
    }
  }

  void prevStep() {
    if (_currentStep > 0) {
      setState(() {
        _currentStep--;
      });
      _pageController.previousPage(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeIn,
      );
    }
  }

  Widget emailInputStep() {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () {
            prevStep(); 
          },
        ),
        title: const Text("Email"),
      ),
      backgroundColor: Colors.white,
      body: buildStepContent(
        "Enter Email",
        Column(
          children: [
            makeInput(label: "Email", controller: _emailController),
            buildNextButton(() {
              _sendEmailOtp();
            }),
          ],
        ),
      ),
    );
  }

  Future<bool> sendEmail(String recipientEmail, String otp) async {
    final smtpServer = gmail(Config.emailUsername, Config.emailPassword);
    final message = Message()
      ..from = Address(Config.emailUsername, 'Your App Name')
      ..recipients.add(recipientEmail)
      ..subject = 'Your OTP for SignUp'
      ..text = 'Your OTP is: $otp';
    try {
      final sendReport = await send(message, smtpServer);
      print('Message sent: ' + sendReport.toString());
      return true;
    } on MailerException catch (e) {
      print('Message not sent. \n' + e.toString());
      return false;
    }
  }

  void _sendEmailOtp() async {
    final email = _emailController.text.trim();
    if (email.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Please enter an email address")),
      );
      return;
    }
    _emailOtp = (100000 + Random().nextInt(900000)).toString();
    bool emailSent = await sendEmail(email, _emailOtp);
    if (emailSent) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("OTP sent to your email")),
      );
      nextStep();
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Failed to send OTP. Please try again.")),
      );
    }
  }

  Widget emailOtpVerificationStep(String title, String subtitle) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () {
            prevStep();
          },
        ),
        title: const Text("OTP Verification"),
      ),
      backgroundColor: Colors.white,
      body: buildStepContent(
        title,
        Column(
          children: [
            Text(
              subtitle,
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.bold,
                color: Colors.black38,
              ),
            ),
            const SizedBox(height: 20),
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: List.generate(6, (index) {
                  return _buildOtpField(index);
                }),
              ),
            ),
            const SizedBox(height: 22),
            buildNextButton(() {
              _verifyEmailOtp();
            }, buttonText: "Verify"),
          ],
        ),
      ),
    );
  }

  void _verifyEmailOtp() {
    String enteredOtp = _otpControllers.map((controller) => controller.text).join();
    if (enteredOtp == _emailOtp) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Email verified successfully")),
      );
      nextStep();
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Invalid OTP. Please try again.")),
      );
    }
  }

  Widget phoneInputStep() {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () {
            prevStep(); 
          },
        ),
        title: const Text("Phone number"),
      ),
      backgroundColor: Colors.white,
      body: buildStepContent(
        "Enter Phone Number",
        Column(
          children: [
            makeInput(label: "Phone Number", controller: _phoneController),
            buildNextButton(() {
              _sendOtp(); 
            }),
          ],
        ),
      ),
    );
  }

  Widget otpVerificationStep() {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () {
            prevStep();
          },
        ),
        title: const Text("OTP Verification"),
      ),
      backgroundColor: Colors.white,
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: List.generate(6, (index) => _buildOtpField(index)),
            ),
            SizedBox(height: 20),
            const SizedBox(height: 22),
            buildNextButton(() {
              _verifyOtp(); 
            }, buttonText: "Verify"),
          ],
        ),
      ),
    );
  }

  Widget _buildOtpField(int index) {
    return SizedBox(
      width: 40,
      child: TextField(
        controller: _otpControllers[index],
        textAlign: TextAlign.center,
        keyboardType: TextInputType.number,
        maxLength: 1,
        onChanged: (value) {
          _onOtpChange(value, index);
        },
        decoration: InputDecoration(
          counterText: "",
          border: OutlineInputBorder(),
        ),
      ),
    );
  }

  String getOtpCode() {
    return _otpControllers.map((controller) => controller.text).join();
  }

  Widget userDetailsStep() {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () {
            prevStep();
          },
        ),
        title: const Text("User Details"),
      ),
      backgroundColor: Colors.white,
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextField(
              controller: _nameController,
              decoration: InputDecoration(labelText: "Full Name"),
            ),
            SizedBox(height: 16),
            TextField(
              controller: _addressController,
              decoration: InputDecoration(labelText: "Address"),
            ),
            SizedBox(height: 16),
            TextField(
              controller: _passwordController,
              decoration: InputDecoration(labelText: "Password"),
              obscureText: true,
            ),
            SizedBox(height: 16),
            TextField(
              controller: _confirmPasswordController,
              decoration: InputDecoration(labelText: "Confirm Password"),
              obscureText: true,
            ),
            SizedBox(height: 24),
            buildNextButton(() {
              _submitUserDetails();
              HomePage();
            }),
          ],
        ),
      ),
    );
  }

  Future<void> _sendOtp() async {
    final String phoneNumber = _phoneController.text.trim();
    _generatedOtp = _generateOtp();
    final response = await http.post(
      Uri.parse('https://rest.nexmo.com/sms/json'),
      headers: <String, String>{
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: {
        'api_key': nexmoApiKey,
        'api_secret': nexmoApiSecret,
        'to': phoneNumber,
        'from': nexmoFromNumber,
        'text': 'Your OTP is: $_generatedOtp',
      },
    );
    if (response.statusCode == 200) {
      final responseData = json.decode(response.body);
      if (responseData['messages'][0]['status'] == '0') {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("OTP sent successfully")),
        );
        _pageController.nextPage(
          duration: Duration(milliseconds: 300),
          curve: Curves.easeInOut,
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Failed to send OTP: ${responseData['messages'][0]['error-text']}")),
        );
      }
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Failed to send OTP. Please try again.")),
      );
    }
  }

  String _generateOtp() {
    return (100000 + Random().nextInt(900000)).toString();
  }

  void _onOtpChange(String value, int index) {
    if (value.isNotEmpty && index < 5) {
      FocusScope.of(context).nextFocus();
    } else if (value.isEmpty && index > 0) {
      FocusScope.of(context).previousFocus();
    }
  }

  void _verifyOtp() {
    String enteredOtp = _otpControllers.map((controller) => controller.text).join();
    if (enteredOtp == _generatedOtp) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("OTP verified successfully")),
      );
      _pageController.nextPage(
        duration: Duration(milliseconds: 300),
        curve: Curves.easeInOut,
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Invalid OTP. Please try again.")),
      );
    }
  }

  void _submitUserDetails() async {
    if (_passwordController.text != _confirmPasswordController.text) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Passwords do not match")),
      );
      return;
    }
    try {
      UserCredential userCredential = await FirebaseAuth.instance.createUserWithEmailAndPassword(
        email: _emailController.text.trim(),
        password: _passwordController.text,
      );
      await FirebaseFirestore.instance.collection('users').doc(userCredential.user?.uid).set({
        'name': _nameController.text,
        'phone': _phoneController.text,
        'email': _emailController.text.trim(),
        'address': _addressController.text,
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("User registered successfully")),
      );
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => HomePage()),
      );
    } on FirebaseAuthException catch (e) {
      String errorMessage = "An error occurred during registration.";
      if (e.code == 'weak-password') {
        errorMessage = "The password provided is too weak.";
      } else if (e.code == 'email-already-in-use') {
        errorMessage = "An account already exists for this email.";
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(errorMessage)),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error during registration: $e")),
      );
    }
  }
  
  Widget makeInput({required String label, required TextEditingController controller, bool obscureText = false}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w400, color: Colors.black87)),
        const SizedBox(height: 5),
        TextField(
          controller: controller,
          obscureText: obscureText,
          decoration: InputDecoration(
            contentPadding: const EdgeInsets.symmetric(vertical: 0, horizontal: 10),
            enabledBorder: OutlineInputBorder(
              borderSide: BorderSide(color: Colors.grey[400]!),
            ),
            border: OutlineInputBorder(
              borderSide: BorderSide(color: Colors.grey[400]!),
            ),
          ),
        ),
        const SizedBox(height: 30),
      ],
    );
  }

  Widget buildStepContent(String title, Widget content) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 40),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(title, style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
          const SizedBox(height: 30),
          content,
        ],
      ),
    );
  }

  Widget buildNextButton(VoidCallback onPressed, {String buttonText = "Next"}) {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton(
        onPressed: onPressed,
        style: ButtonStyle(
          foregroundColor: WidgetStateProperty.all<Color>(Colors.white),
          backgroundColor: WidgetStateProperty.all<Color>(const Color.fromARGB(255, 29, 81, 111)),
          shape: WidgetStateProperty.all<RoundedRectangleBorder>(
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(25.0)),
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.all(15.0),
          child: Text(buttonText, style: const TextStyle(fontSize: 16)),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: PageView(
        controller: _pageController,
        physics: NeverScrollableScrollPhysics(),
        children: [
          emailInputStep(),
          emailOtpVerificationStep("Email Verification", "Enter the OTP sent to your email"),
          phoneInputStep(),
          otpVerificationStep(),
          userDetailsStep(),
        ],
      ),
    );
  }
}
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

class SignupPage extends StatefulWidget {
  const SignupPage({super.key});

  @override
  _SignupPageState createState() => _SignupPageState();
}

class _SignupPageState extends State<SignupPage> {
  final PageController _pageController = PageController(initialPage: 0);

  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _phoneController = TextEditingController();
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _confirmPasswordController = TextEditingController();
  final List<TextEditingController> _otpControllers =
    List.generate(6, (index) => TextEditingController());

  int _currentStep = 0;
  String _verificationId = ''; // Store the verification ID for OTP

  void _onOtpChange(String value, int index) {
    if (value.isNotEmpty && index < 5) {
      FocusScope.of(context).nextFocus(); // Move focus to the next field
    } else if (value.isEmpty && index > 0) {
      FocusScope.of(context).previousFocus(); // Move focus to the previous field
    }
  }

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
      _pageController.previousPage(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeIn,
      );
      setState(() {
        _currentStep--;
      });
    }
  }

  // Back button builder
  Widget buildBackButton() {
    return Align(
      alignment: Alignment.topLeft,
      child: IconButton(
        icon: const Icon(Icons.arrow_back, color: Colors.black),
        onPressed: () {
          prevStep();
        },
      ),
    );
  }

  // Step 1: Email Input
  Widget emailInputStep() {
    return buildStepContent(
      "Enter Email",
      Column(
        children: [
          makeInput(label: "Email", controller: _emailController),
          buildNextButton(() {
            nextStep();
          }),
        ],
      ),
    );
  }

  // Step 2: Phone Input (Trigger OTP)
  Widget phoneInputStep() {
    return buildStepContent(
      "Enter Phone Number",
      Column(
        children: [
          makeInput(label: "Phone Number", controller: _phoneController),
          buildNextButton(() {
            _verifyPhoneNumber(); // Trigger Firebase OTP sending
          }),
        ],
      ),
    );
  }

  // Step 3: OTP Verification Step
  Widget otpVerificationStep(String title, String subtitle) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: buildStepContent(
        title,
        Column(
          children: [
            buildBackButton(), // Back button to go back to the previous step
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
              _verifyOtp(); // Verify the OTP entered by the user
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
        controller: _otpControllers[index], // Use individual controller for each field
        autofocus: index == 0, // Auto-focus the first field
        onChanged: (value) => _onOtpChange(value, index), // Move focus on input change
        showCursor: false,
        readOnly: false,
        textAlign: TextAlign.center,
        style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
        keyboardType: TextInputType.number,
        maxLength: 1,
        decoration: InputDecoration(
          counterText: '',
          enabledBorder: OutlineInputBorder(
            borderSide: const BorderSide(width: 2, color: Colors.black12),
            borderRadius: BorderRadius.circular(12),
          ),
          focusedBorder: OutlineInputBorder(
            borderSide: const BorderSide(width: 2, color: Color.fromARGB(255, 29, 81, 111)),
            borderRadius: BorderRadius.circular(12),
          ),
        ),
      ),
    );
  }

  String getOtpCode() {
    return _otpControllers.map((controller) => controller.text).join();
  }

  // Step 4: User Details (name, password)
  Widget userDetailsStep() {
    return Scaffold(
      resizeToAvoidBottomInset: true,
      backgroundColor: Colors.white,
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20.0),
        child: SingleChildScrollView(
          child: SafeArea(
            child: Column(
              children: [
                buildBackButton(), // Back button to go back to the previous step (OTP verification)
                const SizedBox(height: 20),
                Align(
                  alignment: Alignment.center,
                  child: Text(
                    "User Details",
                    style: const TextStyle(
                      fontSize: 30,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                makeInput(label: "Full Name", controller: _nameController),
                makeInput(label: "Password", controller: _passwordController, obscureText: true),
                makeInput(label: "Confirm Password", controller: _confirmPasswordController, obscureText: true),
                const SizedBox(height: 20),
                buildNextButton(() {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text("Registration Complete!")),
                  );
                }, buttonText: "Register"),
                const SizedBox(height: 40),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // Firebase Phone Authentication: Send OTP
  void _verifyPhoneNumber() async {
    await FirebaseAuth.instance.verifyPhoneNumber(
      phoneNumber: _phoneController.text.trim(),
      verificationCompleted: (PhoneAuthCredential credential) async {
        await FirebaseAuth.instance.signInWithCredential(credential);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Phone number automatically verified")),
        );
      },
      verificationFailed: (FirebaseAuthException e) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Verification failed: ${e.message}")),
        );
      },
      codeSent: (String verificationId, int? resendToken) {
        _verificationId = verificationId;
        nextStep(); // Proceed to OTP step
      },
      codeAutoRetrievalTimeout: (String verificationId) {
        _verificationId = verificationId;
      },
    );
  }

  // Firebase OTP Verification
  void _verifyOtp() async {
    String otpCode = getOtpCode(); // Collect OTP from all boxes
    PhoneAuthCredential credential = PhoneAuthProvider.credential(
      verificationId: _verificationId,
      smsCode: otpCode,
    );

    try {
      await FirebaseAuth.instance.signInWithCredential(credential);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Phone number verified successfully")),
      );
      nextStep(); // Proceed to user details step
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Invalid OTP, try again")),
      );
    }
  }

  // Input Field Builder
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

  // Step Content Wrapper
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

  // Next Button
  Widget buildNextButton(VoidCallback onPressed, {String buttonText = "Next"}) {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton(
        onPressed: onPressed,
        style: ButtonStyle(
          foregroundColor: MaterialStateProperty.all<Color>(Colors.white),
          backgroundColor: MaterialStateProperty.all<Color>(const Color.fromARGB(255, 29, 81, 111)),
          shape: MaterialStateProperty.all<RoundedRectangleBorder>(
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
        physics: const NeverScrollableScrollPhysics(),
        children: [
          emailInputStep(), // Step 1: Email
          phoneInputStep(), // Step 2: Phone input and OTP sending
          otpVerificationStep("Enter OTP", "Enter the 6-digit code sent to your phone"), // Step 3: OTP Verification
          userDetailsStep(), // Step 4: User Details
        ],
      ),
    );
  }
}

import 'package:flutter/material.dart';

class SignupPage extends StatefulWidget {
  const SignupPage({super.key});

  @override
  _SignupPageState createState() => _SignupPageState();
}

class _SignupPageState extends State<SignupPage> {
  final PageController _pageController = PageController(initialPage: 0);

  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _phoneController = TextEditingController();
  final TextEditingController _otpController = TextEditingController();
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _usernameController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _confirmPasswordController = TextEditingController();
  final TextEditingController _addressController = TextEditingController();

  int _currentStep = 0;

  void nextStep() {
    if (_currentStep < 4) {
      _pageController.nextPage(
        duration: const Duration(milliseconds: 300), 
        curve: Curves.easeIn
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
        curve: Curves.easeIn
      );
      setState(() {
        _currentStep--;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: true,
      backgroundColor: Colors.white,
      appBar: AppBar(
        elevation: 0,
        backgroundColor: Colors.white,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, size: 20, color: Colors.black),
          onPressed: prevStep,
        ),
      ),
      body: PageView(
        controller: _pageController,
        physics: const NeverScrollableScrollPhysics(),
        children: [
          emailInputStep(),
          otpVerificationStep("Email OTP", "Enter the OTP sent to your email"),
          phoneInputStep(),
          otpVerificationStep("Phone OTP", "Enter the OTP sent to your phone"),
          userDetailsStep(),
        ],
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
            // Simulate sending email OTP here
            nextStep();
          }),
        ],
      ),
    );
  }

  // Step 3: Phone Input
  Widget phoneInputStep() {
    return buildStepContent(
      "Enter Phone Number",
      Column(
        children: [
          makeInput(label: "Phone Number", controller: _phoneController),
          buildNextButton(() {
            // Simulate sending phone OTP here
            nextStep();
          }),
        ],
      ),
    );
  }

  // Step 5: User Details (name, username, password, address)
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
                const SizedBox(height: 20), // Space at the top
                Align(
                  alignment: Alignment.center, // Center the title horizontally
                  child: Text(
                    "User Details",
                    style: const TextStyle(
                      fontSize: 30,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                const SizedBox(height: 20), // Space below the title
                makeInput(label: "Full Name", controller: _nameController),
                makeInput(label: "Username", controller: _usernameController),
                makeInput(label: "Address", controller: _addressController),
                makeInput(label: "Password", controller: _passwordController, obscureText: true),
                makeInput(label: "Confirm Password", controller: _confirmPasswordController, obscureText: true),
                const SizedBox(height: 20), // Space before the button
                buildNextButton(() {
                  // Trigger final registration process here
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text("Registration Complete!"))
                  );
                }, buttonText: "Register"),
                const SizedBox(height: 40), // Space at the bottom
              ],
            ),
          ),
        ),
      ),
    );
  }

  // Step 2 & 4: OTP Verification Step (Reusing the same design as OtpPage)
  Widget otpVerificationStep(String title, String subtitle) {
    return buildStepContent(
      title,
      Column(
        children: [
          Text(subtitle, style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.bold,
              color: Colors.black38
            )
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
              children: List.generate(
                6,
                (index) =>
                    _textFieldOTP(first: index == 0, last: index == 5),
              ),
            ),
          ),
          const SizedBox(height: 22),
          buildNextButton(() {
            // Simulate OTP verification logic here
            nextStep();
          }, buttonText: "Verify")
        ],
      ),
    );
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
            contentPadding: const EdgeInsets.symmetric(vertical: 0,horizontal: 10),
            enabledBorder: OutlineInputBorder(
              borderSide: BorderSide(color: Colors.grey[400]!)
            ),
            border: OutlineInputBorder(
              borderSide: BorderSide(color: Colors.grey[400]!)
            ),
          ),
        ),
        const SizedBox(height: 30),
      ],
    );
  }

  // OTP Input Fields
  Widget _textFieldOTP({required bool first, last}) {
    return Expanded(
      child: SizedBox(
        height: 50,
        child: TextField(
          autofocus: true,
          onChanged: (value) {
            if (value.length == 1 && last == false) {
              FocusScope.of(context).nextFocus();
            }
            if (value.isEmpty && first == false) {
              FocusScope.of(context).previousFocus();
            }
          },
          showCursor: false,
          readOnly: false,
          textAlign: TextAlign.center,
          style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          keyboardType: TextInputType.number,
          maxLength: 1,
          decoration: InputDecoration(
            counter: const Offstage(),
            enabledBorder: OutlineInputBorder(
                borderSide: const BorderSide(width: 2, color: Colors.black12),
                borderRadius: BorderRadius.circular(12)),
            focusedBorder: OutlineInputBorder(
                borderSide: const BorderSide(
                    width: 2, color: Color.fromARGB(255, 29, 81, 111)),
                borderRadius: BorderRadius.circular(12)),
          ),
        ),
      ),
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
          foregroundColor: WidgetStateProperty.all<Color>(Colors.white),
          backgroundColor: WidgetStateProperty.all<Color>(const Color.fromARGB(255, 29, 81, 111)),
          shape: WidgetStateProperty.all<RoundedRectangleBorder>(
            RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(24.0),
            ),
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.all(14.0),
          child: Text(buttonText, style: const TextStyle(fontSize: 16)),
        ),
      ),
    );
  }
}

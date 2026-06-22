import numpy as np

# --- 1. ضرب المصفوفات (Matrix Multiplication) ---
# لضرب مصفوفتين، نستخدم الرمز @. يجب أن يتوافق عدد أعمدة الأولى مع صفوف الثانية.
A = np.array([[1, 2], [3, 4], [5, 6]])   # مصفوفة 3×2
B = np.array([[7, 8, 9], [10, 11, 12]]) # مصفوفة 2×3
C = A @ B 
print("حاصل ضرب المصفوفات A @ B:\n", C)

# --- 2. الضرب العنصري (Element-wise Multiplication) ---
# ضرب كل عنصر في المقابل له، ويشترط تساوي الأبعاد تماماً.
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("\nالضرب العنصري لمتجهين a * b:", a * b) 

# --- 3. أنواع خاصة من المصفوفات ---
# مصفوفة الهوية (Identity Matrix): القطر الرئيسي 1 والباقي 0.
I = np.eye(3) 
print("\nمصفوفة الوحدة 3x3:\n", I)

# المصفوفة القطرية (Diagonal Matrix): تحديد عناصر القطر الرئيسي.
D = np.diag([2, 5, 3]) 
print("مصفوفة قطرية:\n", D)

# المصفوفة المتماثلة (Symmetric Matrix): التي تساوي مدورها (A = A^T).
A_sym = np.array([[1, 2, 3], [2, 5, 4], [3, 4, 6]])
is_symmetric = np.allclose(A_sym, A_sym.T)
print(f"هل المصفوفة A_sym متماثلة؟ {is_symmetric}")

# --- 4. حساب الزاوية بين متجهين (Angle Between Vectors) ---
u = np.array([2, 6])
v = np.array([2, 5])

dot = np.dot(u, v)                # الضرب القياسي
norm_u = np.linalg.norm(u)        # طول (معيار) المتجه u
norm_v = np.linalg.norm(v)        # طول (معيار) المتجه v

# قانون جيب التمام: cos(theta) = (u . v) / (||u|| * ||v||)
cosin_theta = dot / (norm_u * norm_v)
theta_radians = np.arccos(cosin_theta)      # الزاوية بالراديان
theta_degrees = np.degrees(theta_radians)   # تحويل الزاوية لدرجات
print(f"\nالزاوية بين u و v بالدرجات: {theta_degrees:.2f}°")

# --- 5. حل الأنظمة الخطية (Solving Linear Systems) ---
# لحل نظام معادلات Ax = b (إيجاد قيم المجاهيل x).
# مثال: 
# 2x + y - z = 8
# -3x - y + 2z = -11
# -2x + y + 2z = -3

A_sys = np.array([[2., 1., -1.], 
                  [-3., -1., 2.], 
                  [-2, 1, 2]])
b_sys = np.array([8., -11., -3])

# استخدام linalg.solve لإيجاد الحل مباشرة
result = np.linalg.solve(A_sys, b_sys)
print("\nحل نظام المعادلات (قيم x, y, z):", result)

# التأكد من صحة الحل عن طريق ضرب A في الناتج ومقارنته بـ b
check_solution = np.allclose(A_sys @ result, b_sys)
print(f"هل الحل صحيح؟ {check_solution}")

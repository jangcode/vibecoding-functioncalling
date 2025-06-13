import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      // Handle token expiration
      localStorage.removeItem('token');
      window.location.href = '/auth/login';
    }
    return Promise.reject(error);
  }
);

export const chatApi = {
  sendMessage: async (message: string) => {
    const response = await api.post('/chat', { message });
    return response.data;
  },
};

export const authApi = {
  login: async (email: string, password: string) => {
    const response = await api.post('/auth/login', { email, password });
    return response.data;
  },
  register: async (email: string, password: string, username: string) => {
    const response = await api.post('/auth/register', { email, password, username });
    return response.data;
  },
};

export const paymentApi = {
  createCheckoutSession: async (priceId: string) => {
    const response = await api.post('/payment/create-checkout-session', { priceId });
    return response.data;
  },
  getSubscriptionStatus: async () => {
    const response = await api.get('/payment/subscription-status');
    return response.data;
  },
};
